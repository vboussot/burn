use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, ops::ConvOptions};
use burn_std::Shape;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearViewMut},
};
use cubek::convolution::components::ConvSetupError;

use crate::{
    kernel::{
        into_contiguous,
        matmul::{MatmulStrategy, matmul},
        utils::{address_type, decompose_linear},
    },
    ops::{numeric::empty_device_dtype, swap_dims},
    tensor::CubeTensor,
};

/// The most elements one block of [`conv_winograd`] transforms: 256 MiB of `f32`, well under the
/// buffer a browser's WebGPU device binds.
const WINOGRAD_BLOCK_ELEMENTS: usize = 1 << 26;

/// Perform a convolution by a 3-cubed kernel of stride and dilation one and padding one, or a 5-cubed
/// one of padding two at a tile of 2, in two or three dimensions, by Winograd's F(m^N, r^N) for `tile` = m of 2 or 4: each (m + 2)-cubed input tile,
/// taken every m voxels, and each kernel are carried to a domain of (m + 2)^N points where one
/// matrix multiplication per point combines the channels, and every tile's product is carried back
/// to the m^N output voxels it covers. In three dimensions the multiplications per output voxel per
/// channel pair are 8 at m = 2 and 3.375 at m = 4, where a direct or columns route makes 27; in two,
/// 4 and 2.25 against 9. The transformed domain, which every kernel reads or writes whole, is as many
/// elements per output voxel per channel. F(4, 3) interpolates at 0, 1, -1, 1/2 and -1/2, whose
/// transforms hold a float32 output to about ten parts in a million. A block transforms at most
/// `WINOGRAD_BLOCK_ELEMENTS` elements, its tiles taken in rows along the first spatial axis.
///
/// * `input` - The input feature map, NHWC
/// * `weight` - The weights (filter) applied to each kernel, NHWC
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `tile` - The output voxels per axis one tile covers, 2 or 4
pub fn conv_winograd<const N: usize>(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
    tile: usize,
) -> Result<CubeTensor, ConvSetupError> {
    assert!(tile == 2 || tile == 4, "a Winograd tile covers 2 or 4 output voxels per axis, not {tile}");
    let rank = input.meta.num_dims();
    let kernel = &weight.meta.shape()[1..rank - 1];
    let taps = kernel[0];
    // A 3-cubed kernel at a tile of 2 or 4, a 5-cubed one at a tile of 2: both at most the six points.
    if !(N == 2 || N == 3)
        || options.groups != 1
        || !(taps == 3 || (taps == 5 && tile == 2))
        || kernel.iter().any(|k| *k != taps)
        || options.stride.iter().any(|s| *s != 1)
        || options.dilation.iter().any(|d| *d != 1)
        || options.padding.iter().any(|p| *p != (taps / 2, taps / 2))
    {
        return Err(ConvSetupError::Unknown);
    }

    // The input transform reads each tile's samples of every channel in turn: a buffer laid out
    // channels-last keeps them together, where a channels-first one (a concatenation's output) spreads
    // one tile over as many planes as it has channels.
    let input = into_contiguous(input);
    let dtype = input.dtype;
    // A point carries up to eight times a sample's magnitude at m = 2 and far more at m = 4, and the
    // output transform cancels most of it, which half precision cannot hold: a half-precision
    // convolution reads its input and weight into float32 transforms and writes its output back in
    // its own dtype.
    let transform = match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    };
    let (client, device) = (input.client.clone(), input.device.clone());
    let shape = input.meta.shape();
    let batch_size = shape[0];
    let extents = &shape[1..=N];
    let in_channels = shape[N + 1];
    let out_channels = weight.meta.shape()[0];
    let points = (tile + taps - 1).pow(N as u32);
    let tiles: Vec<usize> = extents.iter().map(|extent| extent.div_ceil(tile)).collect();
    let plane: usize = tiles[1..].iter().product();
    let widest = Ord::max(in_channels, out_channels);
    let budget = Ord::min(
        WINOGRAD_BLOCK_ELEMENTS,
        client.properties().memory.max_page_size as usize / transform.size(),
    );

    // The kernels on the points, [points, out, in], multiplied as [points, in, out].
    let kernels = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([points, out_channels, in_channels]),
        transform,
    );
    let pairs = out_channels * in_channels;
    let cube_dim = CubeDim::new(&client, pairs);
    let cube_count = calculate_cube_count_elemwise(&client, pairs, cube_dim);
    unsafe {
        winograd_kernel_kernel::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            address_type!(weight, kernels),
            weight.into_tensor_arg(),
            kernels.clone().into_linear_view(),
            in_channels,
            pairs,
            N,
            taps,
            tile,
            [dtype_to_storage_type(dtype), dtype_to_storage_type(transform)],
        )
    };
    let kernels = swap_dims(kernels, 1, 2);

    let mut out_shape = shape.to_vec();
    out_shape[N + 1] = out_channels;
    let output = empty_device_dtype(client.clone(), device.clone(), Shape::from(out_shape), dtype);

    // One block for the whole batch when it fits, else blocks of tile rows within each batch entry.
    let blocks: Vec<[usize; 4]> = if points * batch_size * tiles[0] * plane * widest <= budget {
        vec![[0, batch_size, 0, tiles[0]]]
    } else {
        let tile_rows = tiles[0];
        let rows = (budget / (points * plane * widest)).clamp(1, tile_rows);
        (0..batch_size)
            .flat_map(|b| {
                (0..tile_rows)
                    .step_by(rows)
                    .map(move |r| [b, 1, r, Ord::min(rows, tile_rows - r)])
            })
            .collect()
    };
    // The positions of a block, [batches, rows, tiles along each other axis, channels].
    let block_shape = |batches: usize, rows: usize, channels: usize| {
        let mut dims = vec![batches, rows];
        dims.extend_from_slice(&tiles[1..]);
        dims.push(channels);
        block_sequence(&dims)
    };

    for [b, batches, r, rows] in blocks {
        let block_tiles = batches * rows * plane;

        let transformed = empty_device_dtype(
            client.clone(),
            device.clone(),
            Shape::new([points, block_tiles, in_channels]),
            transform,
        );
        let positions = block_tiles * in_channels;
        let cube_dim = CubeDim::new(&client, positions);
        let cube_count = calculate_cube_count_elemwise(&client, positions, cube_dim);
        unsafe {
            winograd_input_kernel::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                address_type!(input, transformed),
                input.clone().into_tensor_arg(),
                transformed.clone().into_linear_view(),
                block_shape(batches, rows, in_channels),
                offsets(b, r),
                positions,
                N,
                taps,
                tile,
                [dtype_to_storage_type(dtype), dtype_to_storage_type(transform)],
            )
        };

        // [points, tiles, in] x [points, in, out] = [points, tiles, out]
        let products = matmul(
            transformed,
            kernels.clone(),
            None,
            MatmulStrategy::default(),
            transform,
        )?;

        let positions = block_tiles * out_channels;
        let cube_dim = CubeDim::new(&client, positions);
        let cube_count = calculate_cube_count_elemwise(&client, positions, cube_dim);
        unsafe {
            winograd_output_kernel::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                address_type!(products, bias, output),
                products.into_tensor_arg(),
                bias.clone().map(|bias| bias.into_buffer_arg()).into(),
                output.clone().into_linear_view(),
                block_shape(batches, rows, out_channels),
                offsets(b, r),
                extents_sequence(extents),
                positions,
                N,
                taps,
                tile,
                [dtype_to_storage_type(dtype), dtype_to_storage_type(transform)],
            )
        };
    }

    Ok(output)
}

fn block_sequence(dims: &[usize]) -> SequenceArg<FastDivmod<usize>> {
    let mut sequence = SequenceArg::new();
    dims.iter().for_each(|dim| sequence.push(*dim));
    sequence
}

fn offsets(batch: usize, row: usize) -> SequenceArg<usize> {
    let mut sequence = SequenceArg::new();
    sequence.push(batch);
    sequence.push(row);
    sequence
}

fn extents_sequence(extents: &[usize]) -> SequenceArg<usize> {
    let mut sequence = SequenceArg::new();
    extents.iter().for_each(|extent| sequence.push(*extent));
    sequence
}

/// `n` to the `power`, at compile time.
fn pow(n: usize, power: usize) -> usize {
    n.pow(power as u32)
}

/// `B^T` in place on the `tile + 2` samples of `x` at `at`, `step` apart, both compile-time so the
/// samples stay in registers. At a tile of two: `[x0 - x2, x1 + x2, x2 - x1, x3 - x1]`; at a tile of
/// four, the interpolation at 0, 1, -1, 1/2, -1/2 and infinity.
#[cube]
fn input_line<E: Float>(
    x: &mut Array<E>,
    #[comptime] at: usize,
    #[comptime] step: usize,
    #[comptime] n: usize,
) {
    if comptime![n == 4] {
        let x0 = x[at];
        let x1 = x[comptime![at + step]];
        let x2 = x[comptime![at + 2 * step]];
        let x3 = x[comptime![at + 3 * step]];
        x[at] = x0 - x2;
        x[comptime![at + step]] = x1 + x2;
        x[comptime![at + 2 * step]] = x2 - x1;
        x[comptime![at + 3 * step]] = x3 - x1;
    } else {
        let x0 = x[at];
        let x1 = x[comptime![at + step]];
        let x2 = x[comptime![at + 2 * step]];
        let x3 = x[comptime![at + 3 * step]];
        let x4 = x[comptime![at + 4 * step]];
        let x5 = x[comptime![at + 5 * step]];
        let quarter = E::new(0.25);
        let half = E::new(0.5);
        x[at] = (x0 - x2 * E::new(5.0)) * quarter + x4;
        x[comptime![at + step]] = (x3 + x4) - (x1 + x2) * quarter;
        x[comptime![at + 2 * step]] = (x4 - x3) + (x1 - x2) * quarter;
        x[comptime![at + 3 * step]] = (x4 - x2) + (x3 - x1) * half;
        x[comptime![at + 4 * step]] = (x4 - x2) + (x1 - x3) * half;
        x[comptime![at + 5 * step]] = (x1 - x3 * E::new(5.0)) * quarter + x5;
    }
}

/// `G` on the `taps` samples of `from` at `at`, `step` apart, written as `n = tile + taps - 1` into
/// `to` at `to_at`, `to_step` apart. Four points, a tile of two on three taps: `[g0, (g0 + g1 + g2) / 2,
/// (g0 - g1 + g2) / 2, g2]`. Six points, at 0, 1, -1, 1/2, -1/2 and infinity, scaled by 4, 2/3, 2/3,
/// -8/3, -8/3 and 1: a tile of four on three taps, or a tile of two on five.
#[cube]
fn kernel_line<E: Float>(
    from: &Array<E>,
    #[comptime] at: usize,
    #[comptime] step: usize,
    to: &mut Array<E>,
    #[comptime] to_at: usize,
    #[comptime] to_step: usize,
    #[comptime] taps: usize,
    #[comptime] n: usize,
) {
    let g0 = from[at];
    let g1 = from[comptime![at + step]];
    let g2 = from[comptime![at + 2 * step]];
    if comptime![n == 4] {
        let half = E::new(0.5);
        to[to_at] = g0;
        to[comptime![to_at + to_step]] = (g0 + g1 + g2) * half;
        to[comptime![to_at + 2 * to_step]] = (g0 - g1 + g2) * half;
        to[comptime![to_at + 3 * to_step]] = g2;
    } else if comptime![taps == 3] {
        let two_thirds = E::new(0.6666667);
        let minus_two_thirds = E::new(-0.6666667);
        let four_g0 = g0 * E::new(4.0);
        let two_g1 = g1 * E::new(2.0);
        to[to_at] = four_g0;
        to[comptime![to_at + to_step]] = (g0 + g1 + g2) * two_thirds;
        to[comptime![to_at + 2 * to_step]] = (g0 - g1 + g2) * two_thirds;
        to[comptime![to_at + 3 * to_step]] = (four_g0 + two_g1 + g2) * minus_two_thirds;
        to[comptime![to_at + 4 * to_step]] = (four_g0 - two_g1 + g2) * minus_two_thirds;
        to[comptime![to_at + 5 * to_step]] = g2;
    } else {
        let g3 = from[comptime![at + 3 * step]];
        let g4 = from[comptime![at + 4 * step]];
        let two_thirds = E::new(0.6666667);
        let minus_two_thirds = E::new(-0.6666667);
        let even = g0 + g2 + g4;
        let odd = g1 + g3;
        // At the half points the taps weigh 1, 1/2, 1/4, 1/8, 1/16, scaled by four.
        let half_even = g0 * E::new(4.0) + g2 + g4 * E::new(0.25);
        let half_odd = g1 * E::new(2.0) + g3 * E::new(0.5);
        to[to_at] = g0 * E::new(4.0);
        to[comptime![to_at + to_step]] = (even + odd) * two_thirds;
        to[comptime![to_at + 2 * to_step]] = (even - odd) * two_thirds;
        to[comptime![to_at + 3 * to_step]] = (half_even + half_odd) * minus_two_thirds;
        to[comptime![to_at + 4 * to_step]] = (half_even - half_odd) * minus_two_thirds;
        to[comptime![to_at + 5 * to_step]] = g4;
    }
}

/// `A^T` on the `n` samples of `from` at `at`, `step` apart, written as `tile` into `to` at `to_at`,
/// `to_step` apart. Four points: `[m0 + m1 + m2, m1 - m2 + m3]`; six points at a tile of two:
/// `[m0 + m1 + m2 + m3 + m4, m1 - m2 + (m3 - m4) / 2 + m5]`.
#[cube]
fn output_line<E: Float>(
    from: &Array<E>,
    #[comptime] at: usize,
    #[comptime] step: usize,
    to: &mut Array<E>,
    #[comptime] to_at: usize,
    #[comptime] to_step: usize,
    #[comptime] tile: usize,
    #[comptime] n: usize,
) {
    let m0 = from[at];
    let m1 = from[comptime![at + step]];
    let m2 = from[comptime![at + 2 * step]];
    let m3 = from[comptime![at + 3 * step]];
    if comptime![n == 4] {
        to[to_at] = m0 + m1 + m2;
        to[comptime![to_at + to_step]] = m1 - m2 + m3;
    } else if comptime![tile == 2] {
        let m4 = from[comptime![at + 4 * step]];
        let m5 = from[comptime![at + 5 * step]];
        to[to_at] = m0 + m1 + m2 + m3 + m4;
        to[comptime![to_at + to_step]] = (m1 - m2) + (m3 - m4) * E::new(0.5) + m5;
    } else {
        let m4 = from[comptime![at + 4 * step]];
        let m5 = from[comptime![at + 5 * step]];
        let even = m1 + m2;
        let odd = m1 - m2;
        let half_even = m3 + m4;
        let half_odd = m3 - m4;
        to[to_at] = m0 + even + half_even;
        to[comptime![to_at + to_step]] = odd + half_odd * E::new(0.5);
        to[comptime![to_at + 2 * to_step]] = even + half_even * E::new(0.25);
        to[comptime![to_at + 3 * to_step]] = odd + half_odd * E::new(0.125) + m5;
    }
}

/// One kernel of the NHWC weight, `(out, in)` = `(ABSOLUTE_POS / in, ABSOLUTE_POS % in)`, carried to
/// the points: `G g G^T` along each of the `dims` axes, one pass per axis from the first, each
/// carrying the 3 taps of every line along that axis to `tile + 2` points, in place of one another
/// between two arrays. The arrays are indexed by constants only, so they stay in registers.
#[cube(launch_unchecked, address_type = "dynamic")]
fn winograd_kernel_kernel<E: Float, T: Float>(
    weight: &Tensor<E>,
    mut points: LinearViewMut<'_, T>,
    in_channels: usize,
    pairs: usize,
    #[comptime] dims: usize,
    #[comptime] taps: usize,
    #[comptime] tile: usize,
    #[define(E, T)] _dtypes: [ElemType; 2],
) {
    if ABSOLUTE_POS >= pairs {
        terminate!();
    }
    let n = comptime![tile + taps - 1];
    let samples = comptime![pow(taps, dims)];
    let count = comptime![pow(n, dims)];
    let out = ABSOLUTE_POS / in_channels;
    let channel = ABSOLUTE_POS % in_channels;
    let base = out * weight.stride(0) + channel * weight.stride(comptime![dims + 1]);

    let mut a = Array::<T>::new(count);
    let mut b = Array::<T>::new(count);
    #[unroll]
    for t in 0..samples {
        let mut at = base;
        #[unroll]
        for axis in 0..dims {
            let coordinate = comptime![(t / pow(taps, dims - 1 - axis)) % taps];
            at += coordinate * weight.stride(comptime![1 + axis]);
        }
        a[t] = T::cast_from(weight[at]);
    }
    // Pass `axis` reads an array whose axes before `axis` are `n` long and from it on `taps` long,
    // and writes one whose axis `axis` is `n` long too.
    #[unroll]
    for axis in 0..dims {
        let step = comptime![pow(taps, dims - 1 - axis)];
        let lines = comptime![pow(n, axis)];
        #[unroll]
        for line in 0..lines {
            #[unroll]
            for offset in 0..step {
                let from = comptime![line * step * taps + offset];
                let to = comptime![line * step * n + offset];
                if comptime![axis % 2 == 0] {
                    kernel_line(&a, from, step, &mut b, to, step, taps, n);
                } else {
                    kernel_line(&b, from, step, &mut a, to, step, taps, n);
                }
            }
        }
    }

    // [points, out, in]: point `p` of pair `ABSOLUTE_POS` at `p * pairs + ABSOLUTE_POS`.
    #[unroll]
    for p in 0..count {
        if comptime![dims % 2 == 1] {
            points.write(p * pairs + ABSOLUTE_POS, b[p]);
        } else {
            points.write(p * pairs + ABSOLUTE_POS, a[p]);
        }
    }
}

/// One tile of one input channel carried to the points: `B^T d B` along each axis. The tile of
/// position `ABSOLUTE_POS / in` of the block reads the input from one voxel before `tile` times its
/// index. Every loop is unrolled so the tile is indexed by constants and stays in registers.
#[cube(launch_unchecked, address_type = "dynamic")]
fn winograd_input_kernel<E: Float, T: Float>(
    input: &Tensor<E>,
    mut transformed: LinearViewMut<'_, T>,
    block_shape: Sequence<FastDivmod<usize>>,
    offsets: Sequence<usize>,
    positions: usize,
    #[comptime] dims: usize,
    #[comptime] taps: usize,
    #[comptime] tile: usize,
    #[define(E, T)] _dtypes: [ElemType; 2],
) {
    if ABSOLUTE_POS >= positions {
        terminate!();
    }
    let n = comptime![tile + taps - 1];
    let count = comptime![pow(n, dims)];
    let pad = comptime![taps / 2];
    let (_, pos) = decompose_linear(ABSOLUTE_POS, &block_shape);
    let batch = pos[0] + offsets[0];
    let channel = pos[dims + 1];
    let base = batch * input.stride(0) + channel * input.stride(comptime![dims + 1]);

    // Coordinates on the input padded by `pad`: a voxel reads the input at `pad` less, inside
    // `pad..pad + extent`. Unsigned, since WebGPU has no 64-bit integer. The samples are read line
    // by line along the last axis, the lines' addresses and bounds settled once each.
    let mut d = Array::<T>::new(count);
    let lines = comptime![count / n];
    let leading = comptime![dims - 1];
    #[unroll]
    for line in 0..lines {
        let mut inside = true;
        let mut at = base;
        #[unroll]
        for axis in 0..leading {
            // A block starts at a row of tiles along the first axis.
            let mut tile_index = pos[1 + axis];
            if comptime![axis == 0] {
                tile_index += offsets[1];
            }
            let coordinate = tile_index * tile + comptime![(line / pow(n, dims - 2 - axis)) % n];
            inside = inside && coordinate >= pad && coordinate < input.shape(comptime![1 + axis]) + pad;
            at += (coordinate - pad) * input.stride(comptime![1 + axis]);
        }
        let mut tile_index = pos[dims];
        if comptime![dims == 1] {
            tile_index += offsets[1];
        }
        let extent = input.shape(dims);
        let stride = input.stride(dims);
        #[unroll]
        for c in 0..n {
            let coordinate = tile_index * tile + c;
            if inside && coordinate >= pad && coordinate < extent + pad {
                d[comptime![line * n + c]] = T::cast_from(input[at + (coordinate - pad) * stride]);
            } else {
                d[comptime![line * n + c]] = T::new(0.0);
            }
        }
    }
    // Along the first axis, then the next: each over every line of the tile along that axis.
    #[unroll]
    for axis in 0..dims {
        let step = comptime![pow(n, dims - 1 - axis)];
        let lines = comptime![pow(n, axis)];
        #[unroll]
        for line in 0..lines {
            #[unroll]
            for offset in 0..step {
                input_line(&mut d, comptime![line * step * n + offset], step, n);
            }
        }
    }

    // [points, tiles, in]: point `p` of position `ABSOLUTE_POS` at `p * positions + ABSOLUTE_POS`.
    #[unroll]
    for p in 0..count {
        transformed.write(p * positions + ABSOLUTE_POS, d[p]);
    }
}

/// One tile of one output channel carried back from the points to the `tile`-cubed voxels it
/// covers, `A^T m A` along each axis, plus the bias; a voxel past the output's extent is not written.
#[cube(launch_unchecked, address_type = "dynamic")]
fn winograd_output_kernel<E: Float, T: Float>(
    products: &Tensor<T>,
    bias: ComptimeOption<&[E]>,
    mut output: LinearViewMut<'_, E>,
    block_shape: Sequence<FastDivmod<usize>>,
    offsets: Sequence<usize>,
    extents: Sequence<usize>,
    positions: usize,
    #[comptime] dims: usize,
    #[comptime] taps: usize,
    #[comptime] tile: usize,
    #[define(E, T)] _dtypes: [ElemType; 2],
) {
    if ABSOLUTE_POS >= positions {
        terminate!();
    }
    let n = comptime![tile + taps - 1];
    let count = comptime![pow(n, dims)];
    let voxels = comptime![pow(tile, dims)];
    let (_, pos) = decompose_linear(ABSOLUTE_POS, &block_shape);
    let batch = pos[0] + offsets[0];
    let channel = pos[dims + 1];
    let out_channels = products.shape(2);
    let tile_index = ABSOLUTE_POS / out_channels;

    let mut a = Array::<T>::new(count);
    let mut b = Array::<T>::new(count);
    let at = tile_index * products.stride(1) + channel * products.stride(2);
    #[unroll]
    for p in 0..count {
        a[p] = products[p * products.stride(0) + at];
    }
    // Pass `axis` reads an array whose axes before `axis` are `tile` long and from it on `n` long,
    // and writes one whose axis `axis` is `tile` long too.
    #[unroll]
    for axis in 0..dims {
        let step = comptime![pow(n, dims - 1 - axis)];
        let lines = comptime![pow(tile, axis)];
        #[unroll]
        for line in 0..lines {
            #[unroll]
            for offset in 0..step {
                let from = comptime![line * step * n + offset];
                let to = comptime![line * step * tile + offset];
                if comptime![axis % 2 == 0] {
                    output_line(&a, from, step, &mut b, to, step, tile, n);
                } else {
                    output_line(&b, from, step, &mut a, to, step, tile, n);
                }
            }
        }
    }

    let bias: ComptimeOption<T> = bias.as_ref().map(|bias| T::cast_from(bias[channel]));
    let bias = bias.unwrap_or_else(|| T::new(0.0));
    #[unroll]
    for v in 0..voxels {
        let mut inside = true;
        let mut at = batch;
        #[unroll]
        for axis in 0..dims {
            let mut tile_index = pos[1 + axis];
            if comptime![axis == 0] {
                tile_index += offsets[1];
            }
            let coordinate = tile_index * tile + comptime![(v / pow(tile, dims - 1 - axis)) % tile];
            let extent = extents[axis];
            inside = inside && coordinate < extent;
            at = at * extent + coordinate;
        }
        if inside {
            if comptime![dims % 2 == 1] {
                output.write(at * out_channels + channel, E::cast_from(b[v] + bias));
            } else {
                output.write(at * out_channels + channel, E::cast_from(a[v] + bias));
            }
        }
    }
}

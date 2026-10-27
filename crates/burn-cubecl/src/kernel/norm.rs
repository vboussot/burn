use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, TensorMetadata};
use burn_std::{Metadata, Shape, strides};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{
        FastDivmod,
        tensor::layout::linear::{LinearView, LinearViewMut},
    },
};

use crate::{
    kernel::{
        cast, into_contiguous,
        utils::{address_type, decompose_linear},
    },
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};

/// The threads the statistics pass spreads the input over, the lanes of every entry together.
const STATISTICS_THREADS: usize = 1 << 17;
/// The partial statistics one thread of a merge pass folds.
const MERGE_SPAN: usize = 64;

/// The mean of each channel of each batch entry of `input` over every axis after the first two, and
/// `sqrt(var + epsilon)` with `var` the biased variance, as two `[batch_size, channels]` tensors of the
/// input's type; a float16 tensor is measured in float32.
pub fn instance_norm_statistics(input: CubeTensor, epsilon: f64) -> (CubeTensor, CubeTensor) {
    let dtype = input.dtype;
    if matches!(dtype, DType::F16 | DType::BF16) {
        let (mean, denominators) = instance_norm_statistics(cast(input, DType::F32), epsilon);
        return (cast(mean, dtype), cast(denominators, dtype));
    }
    let input = dense(input);
    let (mean, denominators) = statistics(input.clone(), epsilon);
    let shape = input.shape();
    let flat = Shape::new([shape[0], shape[1]]);
    let as_flat = |tensor: CubeTensor| {
        CubeTensor::new(
            tensor.client.clone(),
            tensor.handle.clone(),
            Metadata::new(flat.clone(), strides![shape[1], 1]),
            tensor.device.clone(),
            dtype,
        )
    };
    (as_flat(mean), as_flat(denominators))
}

/// `tensor` where it lies when its buffer holds each element once and nothing else, contiguous else.
fn dense(tensor: CubeTensor) -> CubeTensor {
    match memory_order(&tensor) {
        Some(_) => tensor,
        None => into_contiguous(tensor),
    }
}

/// The mean and `sqrt(var + epsilon)` of each entry of the dense `input`, `[entries]` each: Welford's
/// statistics of each lane of every entry, merged by spans until one thread per entry finishes.
fn statistics(input: CubeTensor, epsilon: f64) -> (CubeTensor, CubeTensor) {
    let dtype = input.dtype;
    let shape = input.shape();
    let rank = shape.num_dims();
    assert!(
        rank >= 3,
        "instance_norm takes a tensor of rank 3 or more, got {rank}"
    );
    let order = memory_order(&input).expect("a dense tensor");
    let strides = input.meta.strides().to_vec();
    let (client, device) = (input.client.clone(), input.device.clone());
    let (batch_size, channels) = (shape[0], shape[1]);
    let entries = batch_size * channels;
    let count = shape.num_elements() / entries;

    // The spatial axes in memory order, so that a thread's consecutive positions are its entry's
    // consecutive addresses; the entry is the faster thread index when the channel is the innermost
    // axis, so that a warp reads the channels of one position together.
    let spatial: Vec<usize> = order.iter().copied().filter(|axis| *axis >= 2).collect();
    let entry_fastest = strides[1] < spatial.iter().map(|axis| strides[*axis]).min().unwrap_or(1);
    let lanes = (STATISTICS_THREADS / entries).clamp(1, count);
    let mut spatial_extents = SequenceArg::new();
    let mut spatial_strides = SequenceArg::new();
    for axis in &spatial {
        spatial_extents.push(shape[*axis]);
        spatial_strides.push(strides[*axis]);
    }
    let partial = |per: usize| {
        empty_device_dtype(
            client.clone(),
            device.clone(),
            Shape::new([entries * per]),
            dtype,
        )
    };
    let (mut means, mut deviations, mut counts) = (partial(lanes), partial(lanes), partial(lanes));
    let threads = entries * lanes;
    let cube_dim = CubeDim::new(&client, threads);
    unsafe {
        instance_norm_statistics_kernel::launch_unchecked(
            &client,
            calculate_cube_count_elemwise(&client, threads, cube_dim),
            cube_dim,
            address_type!(input, means, deviations),
            input.clone().into_tensor_arg(),
            means.clone().into_linear_view(),
            deviations.clone().into_linear_view(),
            counts.clone().into_linear_view(),
            SpatialLaunch::new(spatial_extents, spatial_strides),
            if entry_fastest { entries } else { lanes },
            channels,
            count,
            lanes,
            threads,
            entry_fastest,
            dtype_to_storage_type(dtype),
        )
    };

    // The lanes' statistics merged by spans of `MERGE_SPAN` until one thread per entry finishes.
    let mut per = lanes;
    loop {
        let groups = per.div_ceil(MERGE_SPAN);
        let finish = groups == 1;
        let (merged_means, merged_deviations, merged_counts) =
            (partial(groups), partial(groups), partial(groups));
        let threads = entries * groups;
        let cube_dim = CubeDim::new(&client, threads);
        unsafe {
            instance_norm_merge_kernel::launch_unchecked(
                &client,
                calculate_cube_count_elemwise(&client, threads, cube_dim),
                cube_dim,
                address_type!(
                    means,
                    deviations,
                    counts,
                    merged_means,
                    merged_deviations,
                    merged_counts
                ),
                means.into_linear_view(),
                deviations.into_linear_view(),
                counts.into_linear_view(),
                merged_means.clone().into_linear_view(),
                merged_deviations.clone().into_linear_view(),
                merged_counts.clone().into_linear_view(),
                InputScalar::new(epsilon, dtype_to_storage_type(dtype)),
                per,
                MERGE_SPAN,
                groups,
                threads,
                finish,
                dtype_to_storage_type(dtype),
            )
        };
        (means, deviations, counts, per) = (merged_means, merged_deviations, merged_counts, groups);
        if finish {
            break;
        }
    }
    (means, deviations)
}

/// The axes of `tensor` from the outermost to the innermost in its buffer, when the buffer holds each
/// element once and nothing else: sorted by stride, each stride is the product of the extents inside it.
fn memory_order(tensor: &CubeTensor) -> Option<Vec<usize>> {
    let (shape, strides) = (tensor.meta.shape(), tensor.meta.strides());
    let mut order: Vec<usize> = (0..shape.num_dims()).collect();
    order.sort_by_key(|axis| core::cmp::Reverse(strides[*axis]));
    let mut inside = 1;
    for axis in order.iter().rev() {
        if strides[*axis] != inside {
            return None;
        }
        inside *= shape[*axis];
    }
    Some(order)
}

/// The spatial axes in memory order: their extents, to lay a position out over them, and their strides.
#[derive(CubeLaunch, CubeType)]
struct Spatial {
    extents: Sequence<FastDivmod<usize>>,
    strides: Sequence<usize>,
}

/// Where spatial position `position` of an entry at `base` lies.
#[cube]
fn spatial_at(base: usize, position: usize, spatial: &Spatial) -> usize {
    let rank = comptime![spatial.extents.len()];
    let (_, index) = decompose_linear(position, &spatial.extents);
    let mut at = base;
    #[unroll]
    for axis in 0..rank {
        at += index[axis] * spatial.strides[axis];
    }
    at
}

/// Welford's mean, sum of squared deviations and count of one lane of one entry: thread `ABSOLUTE_POS`
/// is `(lane, entry)` with the entry fastest, or `(entry, lane)` with the lane fastest, `inner` being
/// the extent of the faster one. Written at `entry * lanes + lane`.
#[cube(launch_unchecked, address_type = "dynamic")]
fn instance_norm_statistics_kernel<E: Float>(
    input: &Tensor<E>,
    mut means: LinearViewMut<'_, E>,
    mut deviations: LinearViewMut<'_, E>,
    mut counts: LinearViewMut<'_, E>,
    spatial: Spatial,
    inner: FastDivmod<usize>,
    channels: FastDivmod<usize>,
    count: usize,
    lanes: usize,
    threads: usize,
    #[comptime] entry_fastest: bool,
    #[define(E)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= threads {
        terminate!();
    }
    let (outer, fast) = inner.div_mod(ABSOLUTE_POS);
    let entry = if entry_fastest { fast } else { outer };
    let lane = if entry_fastest { outer } else { fast };
    let (batch, channel) = channels.div_mod(entry);
    let base = batch * input.stride(0) + channel * input.stride(1);

    let mut n = 0usize;
    let mut mean = E::new(0.0_f32);
    let mut deviation = E::new(0.0_f32);
    // The lane's positions: `lane`, `lane + lanes`, and so on.
    for i in 0..(count - lane + lanes - 1) / lanes {
        let x = input[spatial_at(base, lane + i * lanes, &spatial)];
        n += 1;
        let delta = x - mean;
        mean += delta / E::cast_from(n);
        deviation += delta * (x - mean);
    }
    means.write(entry * lanes + lane, mean);
    deviations.write(entry * lanes + lane, deviation);
    counts.write(entry * lanes + lane, E::cast_from(n));
}

/// Thread `ABSOLUTE_POS` = `(entry, group)` merges (Chan) the statistics of its entry's partials from
/// `group * span`, `span` of the `per` it has; finishing, it writes `sqrt(var + epsilon)` in place of
/// the sum of squared deviations.
#[cube(launch_unchecked, address_type = "dynamic")]
fn instance_norm_merge_kernel<E: Float>(
    means: LinearView<'_, E>,
    deviations: LinearView<'_, E>,
    counts: LinearView<'_, E>,
    mut merged_means: LinearViewMut<'_, E>,
    mut merged_deviations: LinearViewMut<'_, E>,
    mut merged_counts: LinearViewMut<'_, E>,
    epsilon: InputScalar,
    per: usize,
    span: usize,
    groups: FastDivmod<usize>,
    threads: usize,
    #[comptime] finish: bool,
    #[define(E)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= threads {
        terminate!();
    }
    let (entry, group) = groups.div_mod(ABSOLUTE_POS);
    let first = group * span;
    let last = if first + span < per {
        first + span
    } else {
        per
    };
    let mut count = E::new(0.0_f32);
    let mut mean = E::new(0.0_f32);
    let mut deviation = E::new(0.0_f32);
    for i in first..last {
        let at = entry * per + i;
        let n = counts.read(at);
        let total = count + n;
        let delta = means.read(at) - mean;
        let weight = n / total;
        mean += delta * weight;
        deviation += deviations.read(at) + delta * delta * weight * count;
        count = total;
    }
    merged_means.write(ABSOLUTE_POS, mean);
    merged_counts.write(ABSOLUTE_POS, count);
    if finish {
        merged_deviations.write(
            ABSOLUTE_POS,
            (deviation / count + epsilon.get::<E>()).sqrt(),
        );
    } else {
        merged_deviations.write(ABSOLUTE_POS, deviation);
    }
}

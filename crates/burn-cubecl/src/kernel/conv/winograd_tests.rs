use super::{conv_direct, conv_winograd, conv_winograd_in_blocks};
use crate::{CubeDevice, ops::into_data_sync, tensor::CubeTensor};
use burn_backend::{DType, ops::ConvOptions};
use burn_std::Shape;
use cubecl::prelude::CubeElement;

/// A channels-last tensor of values in `[-1, 1)`.
fn upload(device: &CubeDevice, shape: &[usize], seed: usize, scale: f32) -> CubeTensor {
    let count = shape.iter().product::<usize>();
    let values: Vec<f32> = (0..count)
        .map(|i| (((i * 7 + seed) % 17) as f32 / 8.5 - 1.0) * scale)
        .collect();
    let client = device.client();
    let handle = client.create_from_slice(f32::as_bytes(&values));
    CubeTensor::new_contiguous(
        client,
        device.clone(),
        Shape::from(shape),
        handle,
        DType::F32,
    )
}

/// Winograd at `tile` against the direct convolution, same padding, on `spatial` extents.
fn check<const N: usize>(
    device: &CubeDevice,
    spatial: [usize; N],
    taps: usize,
    tile: usize,
    channels: [usize; 2],
    max_block: usize,
) {
    let [ci, co] = channels;
    let mut in_shape = vec![2];
    in_shape.extend(spatial);
    in_shape.push(ci);
    let mut weight_shape = vec![co];
    weight_shape.extend([taps; N]);
    weight_shape.push(ci);
    let input = upload(device, &in_shape, 0, 1.0);
    // Weights of about 1 / (in * taps^N), so the output stays near unit magnitude.
    let weight = upload(
        device,
        &weight_shape,
        3,
        1.0 / (ci * taps.pow(N as u32)) as f32,
    );
    let bias = upload(device, &[co], 5, 0.5);
    let options = ConvOptions::new([1; N], [taps / 2; N], [1; N], 1);

    let winograd = conv_winograd_in_blocks::<N>(
        input.clone(),
        weight.clone(),
        Some(bias.clone()),
        options.clone(),
        tile,
        max_block,
    )
    .expect("Winograd takes a same-padded convolution of stride and dilation one");
    let direct = conv_direct::<N>(input, weight, Some(bias), options).unwrap();
    assert_eq!(winograd.meta.shape(), direct.meta.shape());

    let actual = into_data_sync(winograd).try_to_vec::<f32>().unwrap();
    let expected = into_data_sync(direct).try_to_vec::<f32>().unwrap();
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-4 + 1e-4 * e.abs(),
            "N={N}, taps={taps}, tile={tile}, index={i}: {a} against {e}"
        );
    }
}

#[test]
fn winograd_matches_the_direct_convolution() {
    let device = CubeDevice::default();
    // Odd extents leave the last tile of each axis overhanging the output.
    for tile in [2, 4] {
        check(&device, [13, 9], 3, tile, [5, 7], usize::MAX);
        check(&device, [7, 9, 11], 3, tile, [5, 7], usize::MAX);
    }
    check(&device, [13, 9], 5, 2, [5, 7], usize::MAX);
    check(&device, [7, 9, 11], 5, 2, [5, 7], usize::MAX);
    // 2 * 64 points * 25 * 24 * 49 tiles * 64 channels pass a block of 2^26 elements: blocks of
    // tile rows within each batch entry.
    check(&device, [97, 49, 47], 3, 2, [64, 2], 1 << 26);
}

#[test]
fn winograd_declines_what_it_does_not_compute() {
    let device = CubeDevice::default();
    let input = upload(&device, &[1, 8, 8, 8, 4], 0, 1.0);
    let weight = upload(&device, &[4, 3, 3, 3, 4], 1, 0.1);
    let decline = |options: ConvOptions<3>, tile| {
        conv_winograd::<3>(input.clone(), weight.clone(), None, options, tile).is_err()
    };
    assert!(decline(ConvOptions::new([2; 3], [1; 3], [1; 3], 1), 2));
    assert!(decline(ConvOptions::new([1; 3], [0; 3], [1; 3], 1), 2));
    assert!(decline(ConvOptions::new([1; 3], [1; 3], [2; 3], 1), 2));
    assert!(decline(ConvOptions::new([1; 3], [1; 3], [1; 3], 2), 2));
}

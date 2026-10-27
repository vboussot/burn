use super::*;
use burn_tensor::module::interpolate3d;
use burn_tensor::ops::{CoordinateTransformMode, InterpolateMode, InterpolateOptions};
use burn_tensor::{Shape, TensorData, Tolerance};

/// `[batch, channels, depth, height, width]` filled with `0, 1, 2, ..`.
fn arange(shape: [usize; 5]) -> TestTensor<5> {
    let device = Default::default();
    TestTensor::from(
        TestTensorInt::arange(0..Shape::new(shape).num_elements() as i64, &device)
            .reshape::<5, _>(shape)
            .into_data(),
    )
}

/// Trilinear interpolation of `arange(shape)` on the host, half-pixel or corners aligned.
fn trilinear(shape: [usize; 5], output: [usize; 3], align_corners: bool) -> TensorData {
    let [batch, channels, depth, height, width] = shape;
    let input = [depth, height, width];
    // The two neighbours and the weight of the second, for each output index along each axis.
    let taps: Vec<Vec<(usize, usize, f64)>> = (0..3)
        .map(|axis| {
            let (size, out) = (input[axis], output[axis]);
            (0..out)
                .map(|i| {
                    let x = if align_corners {
                        i as f64 * (size - 1) as f64 / (out - 1).max(1) as f64
                    } else {
                        ((i as f64 + 0.5) * size as f64 / out as f64 - 0.5).max(0.0)
                    };
                    let low = (x.floor() as usize).min(size - 1);
                    ((low), (low + 1).min(size - 1), x - low as f64)
                })
                .collect()
        })
        .collect();
    let mut values = Vec::new();
    for plane in 0..batch * channels {
        for &(z0, z1, wz) in &taps[0] {
            for &(y0, y1, wy) in &taps[1] {
                for &(x0, x1, wx) in &taps[2] {
                    let at = |z: usize, y: usize, x: usize| {
                        (((plane * depth + z) * height + y) * width + x) as f64
                    };
                    let lerp = |a: f64, b: f64, w: f64| a * (1.0 - w) + b * w;
                    let row = |z, y| lerp(at(z, y, x0), at(z, y, x1), wx);
                    let slice = |z| lerp(row(z, y0), row(z, y1), wy);
                    values.push(lerp(slice(z0), slice(z1), wz) as f32);
                }
            }
        }
    }
    TensorData::new(values, [batch, channels, output[0], output[1], output[2]])
}

#[test]
fn interpolate3d_trilinear_matches_a_host_reference() {
    // Different sizes along every axis, up and down, catch a swapped or skipped axis.
    let shape = [2, 3, 3, 4, 5];
    for (output, align_corners) in [([5, 3, 7], false), ([5, 3, 7], true), ([2, 6, 4], false)] {
        let out = interpolate3d(
            arange(shape),
            Some(output),
            None,
            InterpolateOptions::new(InterpolateMode::Bilinear {
                coordinate_transform: if align_corners {
                    CoordinateTransformMode::AlignCorners
                } else {
                    CoordinateTransformMode::HalfPixel
                },
            }),
        );
        out.into_data().assert_approx_eq::<FloatElem>(
            &trilinear(shape, output, align_corners),
            Tolerance::permissive(),
        );
    }
}

#[test]
fn interpolate3d_nearest_repeats_the_source_pixel() {
    // Doubling along every axis by scale factor repeats each pixel twice per axis.
    let out = interpolate3d(
        arange([1, 1, 2, 2, 2]),
        None,
        Some([2.0, 2.0, 2.0]),
        InterpolateOptions::new(InterpolateMode::Nearest),
    );
    let expected: Vec<f32> = (0..64)
        .map(|i| {
            let (z, y, x) = (i / 16, i / 4 % 4, i % 4);
            (z / 2 * 4 + y / 2 * 2 + x / 2) as f32
        })
        .collect();
    out.into_data()
        .assert_eq(&TensorData::new(expected, [1, 1, 4, 4, 4]), false);

    // Shrinking takes the floor of the scaled coordinate: 3 -> 2 keeps pixels 0 and 1.
    let out = interpolate3d(
        arange([1, 1, 3, 1, 1]),
        Some([2, 1, 1]),
        None,
        InterpolateOptions::new(InterpolateMode::Nearest),
    );
    out.into_data()
        .assert_eq(&TensorData::new(vec![0.0f32, 1.0], [1, 1, 2, 1, 1]), false);
}

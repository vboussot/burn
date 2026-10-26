use super::*;
use burn_tensor::{ElementConversion, TensorData};
use burn_tensor::Tolerance;
use burn_tensor::module::interpolate3d;
use burn_tensor::ops::{InterpolateMode, InterpolateOptions};

fn run(values: Vec<f32>, shape: [usize; 5], output_size: [usize; 3], mode: InterpolateMode, align_corners: bool) -> TensorData {
    let device = Default::default();
    let x = TestTensor::<5>::from_data(TensorData::new(values, shape), &device);
    interpolate3d(x, output_size, InterpolateOptions::new(mode).with_align_corners(align_corners)).into_data()
}

#[test]
fn interpolate3d_nearest_repeats_the_source_pixel() {
    // Doubling along every axis repeats each pixel twice per axis, as PyTorch's nearest does.
    let out = run(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 1, 2, 2, 2], [4, 4, 4], InterpolateMode::Nearest, false);
    let values = out.to_vec::<FloatElem>().unwrap();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let expected = 1.0 + (z / 2 * 4 + y / 2 * 2 + x / 2) as f32;
                assert_eq!(values[(z * 4 + y) * 4 + x], expected.elem::<FloatElem>());
            }
        }
    }
    // Shrinking takes the floor of the scaled coordinate: 3 -> 2 keeps pixels 0 and 1.
    let out = run(vec![1.0, 2.0, 3.0], [1, 1, 1, 1, 3], [1, 1, 2], InterpolateMode::Nearest, false);
    out.assert_eq(&TensorData::new(vec![1.0f32, 2.0], [1, 1, 1, 1, 2]), false);
}

#[test]
fn interpolate3d_linear_blends_the_neighbours_along_each_axis() {
    // Along the last axis, half-pixel: [1, 2] -> [1, 1.25, 1.75, 2]; corners aligned: [1, 4/3, 5/3, 2].
    let out = run(vec![1.0, 2.0], [1, 1, 1, 1, 2], [1, 1, 4], InterpolateMode::Bilinear, false);
    out.assert_approx_eq::<FloatElem>(&TensorData::new(vec![1.0f32, 1.25, 1.75, 2.0], [1, 1, 1, 1, 4]), Tolerance::default());
    let out = run(vec![1.0, 2.0], [1, 1, 1, 1, 2], [1, 1, 4], InterpolateMode::Bilinear, true);
    out.assert_approx_eq::<FloatElem>(&TensorData::new(vec![1.0f32, 4.0 / 3.0, 5.0 / 3.0, 2.0], [1, 1, 1, 1, 4]), Tolerance::default());
    // The same along the first spatial axis; a trilinear centre, half-pixel, is the mean of the
    // eight corners, and with the corners aligned an output of one is the first pixel.
    let out = run(vec![1.0, 2.0], [1, 1, 2, 1, 1], [4, 1, 1], InterpolateMode::Bilinear, false);
    out.assert_approx_eq::<FloatElem>(&TensorData::new(vec![1.0f32, 1.25, 1.75, 2.0], [1, 1, 4, 1, 1]), Tolerance::default());
    let corners: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let out = run(corners.clone(), [1, 1, 2, 2, 2], [1, 1, 1], InterpolateMode::Bilinear, false);
    out.assert_approx_eq::<FloatElem>(&TensorData::new(vec![4.5f32], [1, 1, 1, 1, 1]), Tolerance::default());
    let out = run(corners, [1, 1, 2, 2, 2], [1, 1, 1], InterpolateMode::Bilinear, true);
    out.assert_approx_eq::<FloatElem>(&TensorData::new(vec![1.0f32], [1, 1, 1, 1, 1]), Tolerance::default());
}

use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, FloatDType, module};

#[test]
fn conv3d_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<5>::random([6, 16, 32, 32, 32], Distribution::Default, &device);
    let weight = TestTensor::<5>::random([12, 8, 3, 3, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([12], Distribution::Default, &device);

    let input_ref = TestTensor::<5>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<5>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = burn_tensor::ops::ConvOptions::new([2, 3, 4], [2, 3, 4], [2, 3, 4], 2);

    let output = module::conv3d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv3d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

// Stride (1, 2, 2) over 320 channels: each output sums 8640 products, which drift by a rounding a
// term when summed in half precision.
#[test]
fn conv3d_in_half_precision_over_many_channels_should_match_reference_backend() {
    let device = Default::default();
    let input = TestTensor::random([1, 320, 6, 8, 10], Distribution::Uniform(-1., 1.), &device);
    let weight = TestTensor::random(
        [320, 320, 3, 3, 3],
        Distribution::Uniform(-0.05, 0.05),
        &device,
    );
    half_precision_conv3d_should_match_reference_backend(input, weight, [1, 2, 2]);
}

// A 3x3x3 convolution of 64 channels, which a matrix-multiply route takes where the device has one.
#[test]
fn conv3d_same_in_half_precision_should_match_reference_backend() {
    let device = Default::default();
    let input = TestTensor::random([1, 64, 32, 32, 32], Distribution::Uniform(-1., 1.), &device);
    let weight = TestTensor::random(
        [64, 64, 3, 3, 3],
        Distribution::Uniform(-0.05, 0.05),
        &device,
    );
    half_precision_conv3d_should_match_reference_backend(input, weight, [1, 1, 1]);
}

/// A half-precision convolution against the reference convolving, in float32, the values half
/// precision holds: what is left is the output's own rounding, half a unit in the last place, under
/// a thousandth of the value, and the rounding of a product (CUDA multiplies in TensorFloat-32,
/// whose products keep ten bits).
fn half_precision_conv3d_should_match_reference_backend(
    input: TestTensor<5>,
    weight: TestTensor<5>,
    stride: [usize; 3],
) {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let channels = weight.dims()[0];
    let half = |tensor: TestTensor<5>| tensor.cast(FloatDType::F16);
    let (input, weight) = (half(input), half(weight));
    let bias = TestTensor::<1>::random([channels], Distribution::Uniform(-0.1, 0.1), &device)
        .cast(FloatDType::F16);

    let full = |tensor: &TestTensor<5>| tensor.clone().cast(FloatDType::F32).to_data();
    let input_ref = TestTensor::<5>::from_data(full(&input), &ref_device);
    let weight_ref = TestTensor::<5>::from_data(full(&weight), &ref_device);
    let bias_ref =
        TestTensor::<1>::from_data(bias.clone().cast(FloatDType::F32).to_data(), &ref_device);

    let options = burn_tensor::ops::ConvOptions::new(stride, [1, 1, 1], [1, 1, 1], 1);

    let output = module::conv3d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv3d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .cast(FloatDType::F32)
        .into_data()
        .assert_approx_eq::<f32>(&output_ref.into_data(), Tolerance::rel_abs(1e-3, 5e-4));
}

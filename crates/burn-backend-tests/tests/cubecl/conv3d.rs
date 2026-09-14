use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, module};

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

// Stride, padding and dilation away from their defaults on every axis at once, a kernel that is not
// a cube, on a batch larger than one: the index arithmetic of a gather is only exercised whole when
// they all move together.
#[test]
fn conv3d_every_option_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<5>::random([2, 5, 9, 10, 11], Distribution::Default, &device);
    let weight = TestTensor::<5>::random([7, 5, 3, 2, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([7], Distribution::Default, &device);

    let input_ref = TestTensor::<5>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<5>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = burn_tensor::ops::ConvOptions::new([2, 1, 3], [1, 0, 2], [1, 2, 1], 1);

    let output = module::conv3d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv3d(input_ref, weight_ref, Some(bias_ref), options);

    assert_eq!(output.dims(), [2, 7, 5, 8, 5]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

// The columns of this convolution, 2 * 40 * 32 * 32 output positions of 64 * 27 taps, exceed the
// 2^26 elements one block of the columns route lays out: it runs 37 and 3 rows per batch entry.
#[test]
fn conv3d_columns_over_several_blocks_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<5>::random([2, 64, 40, 32, 32], Distribution::Default, &device);
    let weight = TestTensor::<5>::random([8, 64, 3, 3, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([8], Distribution::Default, &device);

    let input_ref = TestTensor::<5>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<5>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = burn_tensor::ops::ConvOptions::new([1, 1, 1], [1, 1, 1], [1, 1, 1], 1);

    let output = module::conv3d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv3d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

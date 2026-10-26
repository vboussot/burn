//! The operations of a 2-D residual encoder on a batch of sixteen slices, the batch a device route
//! settles on, against the reference backend.

use super::*;
use burn_tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn_tensor::{Distribution, Tolerance, module};

fn pair<const D: usize>(shape: [usize; D]) -> (TestTensor<D>, TestTensor<D>) {
    let tensor = TestTensor::<D>::random(shape, Distribution::Uniform(-1., 1.), &Default::default());
    let reference = TestTensor::<D>::from_data(tensor.to_data(), &ReferenceDevice::new());
    (tensor, reference)
}

fn assert_close<const D: usize>(output: TestTensor<D>, reference: TestTensor<D>) {
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&reference.into_data(), Tolerance::rel_abs(1e-3, 1e-4));
}

fn conv2d_case(input: [usize; 4], weight: [usize; 4], stride: usize, padding: usize) {
    let (x, x_ref) = pair(input);
    let (w, w_ref) = pair(weight);
    let options = ConvOptions::new([stride, stride], [padding, padding], [1, 1], 1);
    assert_close(
        module::conv2d(x, w, None, options.clone()),
        module::conv2d(x_ref, w_ref, None, options),
    );
}

#[test]
fn batch_of_slices_conv2d_same_padding() {
    conv2d_case([16, 24, 256, 256], [24, 24, 3, 3], 1, 1);
}

#[test]
fn batch_of_slices_conv2d_first_layer() {
    conv2d_case([16, 5, 256, 256], [24, 5, 3, 3], 1, 1);
}

#[test]
fn batch_of_slices_conv2d_strided() {
    conv2d_case([16, 24, 256, 256], [48, 24, 3, 3], 2, 1);
}

#[test]
fn batch_of_slices_conv2d_pointwise() {
    conv2d_case([16, 24, 128, 128], [48, 24, 1, 1], 1, 0);
}

#[test]
fn batch_of_slices_conv_transpose2d() {
    let (x, x_ref) = pair([16, 48, 128, 128]);
    let (w, w_ref) = pair([48, 24, 2, 2]);
    let options = ConvTransposeOptions::new([2, 2], [0, 0], [0, 0], [1, 1], 1);
    assert_close(
        module::conv_transpose2d(x, w, None, options.clone()),
        module::conv_transpose2d(x_ref, w_ref, None, options),
    );
}

// The product a Winograd point makes over 24 input channels: an inner extent no multiple of a unit
// matmul's tile, on a batch.
#[test]
fn batch_of_slices_matmul_over_an_inner_extent_of_24() {
    let (lhs, lhs_ref) = pair([16, 1024, 24]);
    let (rhs, rhs_ref) = pair([16, 24, 32]);
    assert_close(lhs.matmul(rhs), lhs_ref.matmul(rhs_ref));
}

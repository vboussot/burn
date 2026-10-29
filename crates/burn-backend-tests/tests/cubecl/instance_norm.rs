use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, module};

fn assert_matches_reference<const D: usize>(
    input: TestTensor<D>,
    reference: TestTensor<D>,
    affine: bool,
) {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let channels = input.dims()[1];
    let gamma = TestTensor::<1>::random([channels], Distribution::Default, &device);
    let beta = TestTensor::<1>::random([channels], Distribution::Default, &device);
    let gamma_ref = TestTensor::<1>::from_data(gamma.to_data(), &ref_device);
    let beta_ref = TestTensor::<1>::from_data(beta.to_data(), &ref_device);
    let (gamma, beta, gamma_ref, beta_ref) = match affine {
        true => (Some(gamma), Some(beta), Some(gamma_ref), Some(beta_ref)),
        false => (None, None, None, None),
    };

    let output = module::instance_norm(input, gamma, beta, 1e-5);
    let output_ref = module::instance_norm(reference, gamma_ref, beta_ref, 1e-5);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

fn on_reference<const D: usize>(tensor: &TestTensor<D>) -> TestTensor<D> {
    TestTensor::from_data(tensor.to_data(), &ReferenceDevice::new())
}

// Many lines per channel (32 * 20 * 24 of 28 elements) on a batch of two.
#[test]
fn instance_norm_should_match_reference_backend() {
    let input = TestTensor::<5>::random(
        [2, 32, 20, 24, 28],
        Distribution::Default,
        &Default::default(),
    );
    let reference = on_reference(&input);
    assert_matches_reference(input, reference, true);
}

#[test]
fn instance_norm_of_rank_three_should_match_reference_backend() {
    let input = TestTensor::<3>::random([3, 5, 17], Distribution::Default, &Default::default());
    let reference = on_reference(&input);
    assert_matches_reference(input, reference, false);
}

// A convolution's output: a channels-last buffer read through channels-first strides.
#[test]
fn instance_norm_of_a_permuted_layout_should_match_reference_backend() {
    let input =
        TestTensor::<5>::random([2, 6, 7, 8, 5], Distribution::Default, &Default::default());
    let reference = on_reference(&input).permute([0, 4, 1, 2, 3]);
    assert_matches_reference(input.permute([0, 4, 1, 2, 3]), reference, true);
}

// Spatial axes out of their buffer order: no single stride walks a channel.
#[test]
fn instance_norm_of_swapped_spatial_axes_should_match_reference_backend() {
    let input =
        TestTensor::<5>::random([2, 3, 5, 6, 7], Distribution::Default, &Default::default());
    let reference = on_reference(&input).permute([0, 1, 3, 2, 4]);
    assert_matches_reference(input.permute([0, 1, 3, 2, 4]), reference, true);
}

// A slice leaves the buffer's other elements around the tensor's.
#[test]
fn instance_norm_of_a_slice_should_match_reference_backend() {
    let input =
        TestTensor::<5>::random([2, 4, 9, 6, 7], Distribution::Default, &Default::default());
    let reference = on_reference(&input).slice([0..2, 1..4, 2..8, 0..6, 1..7]);
    assert_matches_reference(input.slice([0..2, 1..4, 2..8, 0..6, 1..7]), reference, true);
}

// The output is written over its input only when no other tensor holds the input's buffer.
#[test]
fn instance_norm_should_leave_a_shared_input_untouched() {
    let input =
        TestTensor::<5>::random([1, 3, 4, 5, 6], Distribution::Default, &Default::default());
    let before = input.to_data();

    let output = module::instance_norm(input.clone(), None, None, 1e-5);
    let _ = output.into_data();

    input.into_data().assert_eq(&before, true);
}

// A batch of sixteen 2-D slices, nnU-Net's residual encoder at the batch a device route settles on:
// 384 entries of 65536 samples, channels first and channels last.
#[test]
fn instance_norm_of_a_large_batch_of_slices_should_match_reference_backend() {
    let input = TestTensor::<4>::random(
        [16, 24, 256, 256],
        Distribution::Default,
        &Default::default(),
    );
    let reference = on_reference(&input);
    assert_matches_reference(input, reference, true);
}

#[test]
fn instance_norm_of_a_large_batch_of_channels_last_slices_should_match_reference_backend() {
    let input = TestTensor::<4>::random(
        [16, 128, 96, 24],
        Distribution::Default,
        &Default::default(),
    );
    let reference = on_reference(&input).permute([0, 3, 1, 2]);
    assert_matches_reference(input.permute([0, 3, 1, 2]), reference, true);
}

// Deeper stages: many entries of few samples (a lane covers several entries' worth of threads).
#[test]
fn instance_norm_of_many_small_entries_should_match_reference_backend() {
    let input =
        TestTensor::<4>::random([16, 256, 8, 8], Distribution::Default, &Default::default());
    let reference = on_reference(&input);
    assert_matches_reference(input, reference, true);
}

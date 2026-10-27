use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::module::instance_norm;

// Float16 reaches the third decimal of these values.
fn tolerance() -> Tolerance<FloatElem> {
    Tolerance::rel_abs(5e-3, 1e-3)
}

fn input() -> TestTensor<4> {
    TestTensor::from([
        [
            [[0.5, -1.2, 3.0], [2.2, 0.1, -0.7]],
            [[-2.0, 4.5, 1.25], [0.0, -3.5, 2.75]],
        ],
        [
            [[1.0, 1.5, -0.5], [0.25, 2.0, -1.0]],
            [[3.0, -1.0, 0.5], [1.5, 2.5, -2.5]],
        ],
    ])
}

#[test]
fn test_instance_norm_with_scale_and_bias() {
    let gamma = TestTensor::<1>::from([2.0, -0.5]);
    let beta = TestTensor::<1>::from([0.1, 0.3]);

    let output = instance_norm(input(), Some(gamma), Some(beta), 1e-5);

    let expected = TensorData::from([
        [
            [[-0.10004, -2.36712, 3.23391], [2.16704, -0.63347, -1.70033]],
            [[0.76094, -0.4375, 0.16172], [0.39219, 1.0375, -0.11484]],
        ],
        [
            [[0.96092, 1.90011, -1.85664], [-0.44786, 2.8393, -2.79583]],
            [[-0.30471, 0.73193, 0.34319], [0.08403, -0.17513, 1.12067]],
        ],
    ]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance());
}

#[test]
fn test_instance_norm_bias_without_scale() {
    let beta = TestTensor::<1>::from([0.1, 0.3]);

    let output = instance_norm(input(), None, Some(beta), 1e-5);

    let expected = TensorData::from([
        [
            [[-0.00002, -1.13356, 1.66695], [1.13352, -0.26673, -0.80016]],
            [[-0.62188, 1.775, 0.57656], [0.11562, -1.175, 1.12969]],
        ],
        [
            [[0.53046, 1.00006, -0.87832], [-0.17393, 1.46965, -1.34792]],
            [[1.50941, -0.56387, 0.21361], [0.73193, 1.25025, -1.34135]],
        ],
    ]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance());
}

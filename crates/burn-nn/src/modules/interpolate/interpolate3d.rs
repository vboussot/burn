use alloc::format;

use burn::tensor::module::{interpolate3d, interpolate3d_output_size};

use crate::modules::bounds::{self, Windowing};
use burn::instrument::{Bounds, compute};
use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::ops::{
    CoordinateTransformMode, CubicOptions, InterpolateOptions, NearestRoundingMode,
};

use super::InterpolateMode;

/// Configuration for the 3D interpolation module.
///
/// This struct defines the configuration options for the 3D interpolation operation.
/// It allows specifying the output size, scale factor, and interpolation mode.
#[derive(Config, Debug)]
pub struct Interpolate3dConfig {
    /// Output size of the interpolated tensor, `[depth, height, width]`.
    /// Exactly one of `output_size` or `scale_factor` must be set.
    #[config(default = "None")]
    pub output_size: Option<[usize; 3]>,

    /// Scale factor for resizing the input tensor, `[depth, height, width]`.
    /// The output size is `floor(input_size * scale_factor)`.
    /// Exactly one of `output_size` or `scale_factor` must be set.
    #[config(default = "None")]
    pub scale_factor: Option<[f32; 3]>,

    /// Interpolation mode to use for resizing.
    /// Determines how the output values are calculated.
    #[config(default = "InterpolateMode::Nearest")]
    pub mode: InterpolateMode,

    /// If `true`, the input and output tensors are aligned by their corner pixels.
    /// If `false`, half-pixel coordinate mapping is used instead.
    #[config(default = true)]
    pub align_corners: bool,

    /// Coordinate transformation mode. `TfCropAndResize` is not supported in 3D.
    #[config(default = "None")]
    pub coordinate_transform: Option<CoordinateTransformMode>,

    /// Nearest rounding mode.
    #[config(default = "None")]
    pub nearest_mode: Option<NearestRoundingMode>,

    /// Cubic interpolation options.
    #[config(default = "None")]
    pub cubic_options: Option<CubicOptions>,
}

/// Interpolate module for resizing tensors with shape [N, C, D, H, W].
///
/// The module resizes the depth, height and width of a volume, preserving the
/// number of channels and batch size, with the modes of [Interpolate2d](super::Interpolate2d).
///
/// The module can be created using the [Interpolate3dConfig] struct and the
/// `init` method, which returns an instance of the [Interpolate3d] struct.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Interpolate3d {
    /// Output size of the interpolated tensor
    pub output_size: Option<[usize; 3]>,

    /// Scale factor for resizing the input tensor
    pub scale_factor: Option<[f32; 3]>,

    /// Interpolation mode used for resizing
    #[module(skip)]
    pub mode: InterpolateMode,

    /// Whether to align corner pixels
    pub align_corners: bool,

    /// Coordinate transformation mode
    #[module(skip)]
    pub coordinate_transform: Option<CoordinateTransformMode>,

    /// Nearest rounding mode
    #[module(skip)]
    pub nearest_mode: Option<NearestRoundingMode>,

    /// Cubic interpolation options
    #[module(skip)]
    pub cubic_options: Option<CubicOptions>,
}

impl Interpolate3dConfig {
    /// Initialize the interpolation module
    pub fn init(self) -> Interpolate3d {
        Interpolate3d {
            output_size: self.output_size,
            scale_factor: self.scale_factor,
            mode: self.mode,
            align_corners: self.align_corners,
            coordinate_transform: self.coordinate_transform,
            nearest_mode: self.nearest_mode,
            cubic_options: self.cubic_options,
        }
    }
}

impl Interpolate3d {
    /// The options the interpolation runs with, without its size.
    fn options(&self) -> InterpolateOptions {
        let mut options = InterpolateOptions::new(self.mode.clone().into());
        if let Some(ct) = self.coordinate_transform {
            options = options.with_coordinate_transform(ct);
        } else {
            options = options.with_align_corners(self.align_corners);
        }
        if let Some(nm) = self.nearest_mode {
            options = options.with_nearest_mode(nm);
        }
        if let Some(co) = self.cubic_options {
            options = options.with_cubic_options(co);
        }
        options
    }

    /// The floor for an interpolation alone: a resampling to the size it produces. Each output
    /// element blends a few neighbours, below this model's resolution, and reads at least one
    /// position of its own along each dimension a downsampling shrinks.
    fn bounds(&self, input: &Tensor<5>) -> Bounds {
        let input_size = bounds::spatial(input);
        let output_size = interpolate3d_output_size(
            input_size,
            self.output_size,
            self.scale_factor,
            self.options(),
        );
        Windowing {
            output_size,
            read_size: core::array::from_fn(|i| input_size[i].min(output_size[i])),
        }
        .resampling(input)
    }

    /// Performs the forward pass of the interpolation module
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape [N, C, D, H, W]
    ///
    /// # Returns
    ///
    /// Resized tensor with shape [N, C, D', H', W'], where D', H' and W' are determined by
    /// the output_size or scale_factor specified in the module configuration
    ///
    /// # Panics
    ///
    /// Panics unless exactly one of `output_size` or `scale_factor` is set.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let input = Tensor::<5>::random([1, 3, 16, 16, 16], Distribution::Uniform(0.0, 1.0), &device);
    /// let interpolate = Interpolate3dConfig::new()
    ///     .with_scale_factor(Some([2.0, 2.0, 2.0]))
    ///     .init();
    /// let output = interpolate.forward(input);
    /// assert_eq!(output.dims(), [1, 3, 32, 32, 32]);
    /// ```
    #[compute(bounds = Self::bounds)]
    pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
        interpolate3d(input, self.output_size, self.scale_factor, self.options())
    }
}

impl ModuleDisplay for Interpolate3d {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add_debug_attribute("mode", &self.mode)
            .add("output_size", &format!("{:?}", self.output_size))
            .add("scale_factor", &format!("{:?}", self.scale_factor))
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;

    #[test]
    fn test_module() {
        let input = Tensor::<5>::random(
            [2, 3, 4, 5, 6],
            Distribution::Uniform(0.0, 1.0),
            &Default::default(),
        );

        let config = Interpolate3dConfig::new().with_output_size(Some([8, 7, 3]));
        let output = config.init().forward(input.clone());
        assert_eq!(output.dims(), [2, 3, 8, 7, 3]);

        // Different scales per axis catch swapped axes.
        let config = Interpolate3dConfig::new()
            .with_scale_factor(Some([0.5, 2.0, 1.5]))
            .with_mode(InterpolateMode::Linear)
            .with_align_corners(false);
        let output = config.init().forward(input);
        assert_eq!(output.dims(), [2, 3, 2, 10, 9]);
    }

    #[test]
    fn display() {
        let config = Interpolate3dConfig::new().with_output_size(Some([20, 20, 20]));
        let layer = config.init();

        assert_eq!(
            alloc::format!("{layer}"),
            "Interpolate3d {mode: Nearest, output_size: Some([20, 20, 20]), \
            scale_factor: None}"
        );
    }
}

use alloc::format;
use burn::tensor::module::interpolate3d;
use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::ops::InterpolateOptions;

use super::InterpolateMode;

/// Configuration for the 3D interpolation module.
///
/// The interpolation is by output size or by scale factor, one of the two.
#[derive(Config, Debug)]
pub struct Interpolate3dConfig {
    /// Output size of the interpolated tensor
    #[config(default = "None")]
    pub output_size: Option<[usize; 3]>,

    /// Scale factor for resizing the input tensor
    #[config(default = "None")]
    pub scale_factor: Option<[f32; 3]>,

    /// Interpolation mode
    #[config(default = "InterpolateMode::Nearest")]
    pub mode: InterpolateMode,

    /// Whether the input and output are aligned by their corner pixels
    #[config(default = true)]
    pub align_corners: bool,
}

/// Interpolate module for resizing 3D tensors with shape [N, C, D, H, W].
///
/// Should be created with [Interpolate3dConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Interpolate3d {
    /// Output size of the interpolated tensor
    pub output_size: Option<[usize; 3]>,

    /// Scale factor for resizing the input tensor
    pub scale_factor: Option<[f32; 3]>,

    /// Interpolation mode
    #[module(skip)]
    pub mode: InterpolateMode,

    /// Whether the input and output are aligned by their corner pixels
    pub align_corners: bool,
}

impl Interpolate3dConfig {
    /// Initialize a new [interpolate 3d](Interpolate3d) module.
    pub fn init(self) -> Interpolate3d {
        Interpolate3d {
            output_size: self.output_size,
            scale_factor: self.scale_factor,
            mode: self.mode,
            align_corners: self.align_corners,
        }
    }
}

impl Interpolate3d {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, channels, depth_in, height_in, width_in]`
    /// - output: `[batch_size, channels, depth_out, height_out, width_out]`
    pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
        let output_size = calculate_output_size(input.dims(), self.output_size, self.scale_factor);
        interpolate3d(
            input,
            output_size,
            InterpolateOptions::new(self.mode.clone().into())
                .with_align_corners(self.align_corners),
        )
    }
}

/// The output size from the given output size, or from the scale factor over the input's extents.
fn calculate_output_size(
    input_dims: [usize; 5],
    output_size: Option<[usize; 3]>,
    scale_factor: Option<[f32; 3]>,
) -> [usize; 3] {
    match (output_size, scale_factor) {
        (Some(output_size), None) => output_size,
        (None, Some(scale_factor)) => {
            let [_, _, d, h, w] = input_dims;
            let scaled = |extent: usize, scale: f32| {
                let scaled = (extent as f64) * (scale as f64);
                if scaled > usize::MAX as f64 {
                    panic!("Scale factor {scale} over {extent} is too large");
                }
                scaled as usize
            };
            [
                scaled(d, scale_factor[0]),
                scaled(h, scale_factor[1]),
                scaled(w, scale_factor[2]),
            ]
        }
        _ => panic!("Either output_size or scale_factor must be provided"),
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
            .add("mode", &format!("{:?}", &self.mode))
            .add("output_size", &format!("{:?}", &self.output_size))
            .add("scale_factor", &format!("{:?}", &self.scale_factor))
            .add("align_corners", &self.align_corners)
            .optional()
    }
}

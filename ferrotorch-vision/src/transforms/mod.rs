pub mod center_crop;
pub mod resize;
pub mod to_tensor;
pub mod vision_normalize;

pub use center_crop::CenterCrop;
pub use resize::Resize;
pub use to_tensor::VisionToTensor;
pub use vision_normalize::VisionNormalize;

/// ImageNet channel-wise means (RGB order), used for input normalization.
pub const IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406];

/// ImageNet channel-wise standard deviations (RGB order).
pub const IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225];

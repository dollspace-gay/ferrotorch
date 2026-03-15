# Feature: Phase 5 — Vision (ferrotorch-vision)

## Summary
A torchvision-equivalent crate providing pre-built model architectures (ResNet, VGG, EfficientNet, ViT, ConvNeXt, Swin Transformer, U-Net, YOLO), standard image datasets (MNIST, CIFAR-10/100, ImageNet loader), image I/O utilities, and image-specific transforms (Resize, CenterCrop, ColorJitter, MixUp, CutMix, and more). Every model implements the `Module` trait from ferrotorch-nn, every dataset implements the `Dataset` trait from ferrotorch-data, and every transform implements the `Transform` trait from ferrotorch-data. Models support pretrained ImageNet weights downloaded from a CDN in SafeTensors format and intermediate feature extraction for transfer learning. A model registry enables dynamic model construction by name. This crate is the "batteries included" layer that lets users train and evaluate vision models out of the box without hand-rolling architectures or data pipelines.

## Requirements
- REQ-1: The crate must provide ResNet variants (18, 34, 50, 101, 152) as structs implementing `Module<T>`. Each variant must accept a `ModelConfig` with a `pretrained: bool` field (loads ImageNet-trained weights from a CDN in SafeTensors format via ferrotorch-serialize when true, random Kaiming initialization when false) and a configurable `num_classes` parameter for the final classification head. When `pretrained` is true and `num_classes` differs from the original 1000, the classification head is re-initialized with random weights while all backbone weights are loaded from the pretrained checkpoint.
- REQ-2: The crate must provide VGG (11, 13, 16, 19), EfficientNet (B0 through B7), Vision Transformer (ViT-B/16, ViT-L/16, ViT-H/14), ConvNeXt (Tiny, Small, Base, Large), Swin Transformer (Tiny, Small, Base, Large), U-Net (encoder-decoder with skip connections), and YOLO (single-shot detection head with anchor-based output) — all implementing `Module<T>` with configurable `num_classes` and optional pretrained ImageNet weight loading following the same `pretrained: bool` pattern as ResNet.
- REQ-3: The crate must provide dataset structs for MNIST (60K train / 10K test, 28x28 grayscale), CIFAR-10 (50K train / 10K test, 32x32 RGB, 10 classes), and CIFAR-100 (50K train / 10K test, 32x32 RGB, 100 classes), each implementing `Dataset` from ferrotorch-data. Datasets must auto-download from canonical URLs to a configurable cache directory (`~/.ferrotorch/datasets/` by default) and verify integrity via SHA-256 checksums before extraction.
- REQ-4: The crate must provide an ImageNet loader struct implementing `Dataset` that reads from a local directory in the standard `train/<class_name>/image.JPEG` layout. It must not attempt to download ImageNet (license restrictions). It must support a `Transform` pipeline applied lazily per sample.
- REQ-5: The crate must provide image-specific transforms: `Resize` (bilinear interpolation to target H x W), `CenterCrop` (extract center region), `RandomCrop` (extract random region with optional padding), `RandomResizedCrop` (crop to random size and aspect ratio, then resize — the standard training crop), `RandomHorizontalFlip` (probability-based), `RandomAffine` (random rotation, translation, scale, shear), `ColorJitter` (random adjustments to brightness, contrast, saturation, hue), `RandomErasing` (randomly erase a rectangular region of the tensor with random values or mean), `Normalize` (per-channel mean/std), and `ToTensor` (convert raw bytes to `Tensor<T>` with shape `[C, H, W]` and values in `[0.0, 1.0]`). All transforms must implement `Transform` from ferrotorch-data.
- REQ-5a: The crate must provide batch-level augmentation transforms: `MixUp` (linear interpolation of two samples and their labels with a Beta-distributed mixing coefficient) and `CutMix` (replace a random rectangular region of one sample with the corresponding region from another, mixing labels proportionally to the area). Both must accept a `DataLoader` batch (tensor + labels) and return the augmented batch, and must implement `BatchTransform` from ferrotorch-data.
- REQ-6: All public functions must return `Result<T, FerrotorchError>`. Invalid inputs (wrong spatial dimensions, out-of-range jitter parameters, missing dataset files, corrupted downloads) must produce descriptive errors, never panics.
- REQ-7: Every model must expose a `forward` method matching the `Module` trait signature. Models that produce non-classification outputs (U-Net segmentation maps, YOLO detection tensors) must document their output tensor shapes in their struct-level rustdoc.
- REQ-8: The crate must provide convenience constructors following the pattern `resnet18(num_classes: usize) -> Result<ResNet<T>, FerrotorchError>` and `resnet18_pretrained(num_classes: usize) -> Result<ResNet<T>, FerrotorchError>` for every model family. Pretrained weight files must use the SafeTensors format via ferrotorch-serialize. All pretrained constructors download ImageNet-trained weights from a CDN (configurable via `FERROTORCH_WEIGHTS_URL` environment variable, defaulting to a hardcoded base URL). Weights are cached at `~/.ferrotorch/weights/{model_name}.safetensors` and verified via SHA-256 checksum before loading.
- REQ-9: Every model must support **feature extraction** — returning intermediate layer activations (feature maps) in addition to the final output. Each model must provide a `forward_features(&self, input: &Tensor<T>, layers: &[&str]) -> Result<HashMap<String, Tensor<T>>, FerrotorchError>` method that returns a map of named intermediate tensors at the requested layer names. A convenience function `create_feature_extractor(model, return_nodes: &[&str]) -> FeatureExtractor<T>` must wrap any model and produce a `Module<T>` whose `forward` returns the feature map dict. This is essential for transfer learning, FPN-based detection, and style transfer.
- REQ-10: The crate must provide a **model registry** with the following API: `models::list_models() -> Vec<&'static str>` returns all available model names (e.g., `"resnet50"`, `"convnext_tiny"`, `"swin_base"`). `models::get_model(name: &str, pretrained: bool, num_classes: usize) -> Result<Box<dyn Module<T>>, FerrotorchError>` constructs and returns the named model. Unknown model names return `Err(FerrotorchError::InvalidArgument { .. })`. The registry must be extensible — users can register custom models via `models::register_model(name, constructor_fn)`.
- REQ-11: The crate must provide image I/O functions: `io::read_image(path: impl AsRef<Path>) -> Result<RawImage, FerrotorchError>` reads a PNG or JPEG file and returns a `RawImage` struct, and `io::write_image(path: impl AsRef<Path>, image: &RawImage) -> Result<(), FerrotorchError>` writes a `RawImage` to disk as PNG or JPEG (format inferred from file extension). These use the `image` crate internally. A `io::read_image_as_tensor(path: impl AsRef<Path>) -> Result<Tensor<T>, FerrotorchError>` convenience function combines reading and `ToTensor` conversion.

## Acceptance Criteria
- [ ] AC-1: `resnet18::<f32>(1000)` constructs a ResNet-18 with random weights. A forward pass on a `Tensor<f32>` of shape `[1, 3, 224, 224]` returns a tensor of shape `[1, 1000]` without error. The same holds for ResNet-34/50/101/152.
- [ ] AC-2: `VGG::vgg16(1000)`, `EfficientNet::b0(1000)`, `ViT::vit_b_16(1000)`, `ConvNeXt::tiny(1000)`, and `SwinTransformer::tiny(1000)` each accept a `[1, 3, 224, 224]` input and produce a `[1, 1000]` output. `UNet::new(3, 21)` accepts `[1, 3, 256, 256]` and produces `[1, 21, 256, 256]`. `Yolo::new(80)` accepts `[1, 3, 416, 416]` and produces a detection tensor whose shape is documented and correct.
- [ ] AC-3: `Mnist::new(root, Split::Train, Some(transform))` downloads MNIST to `root` (or uses the cached copy), verifies SHA-256 checksums, and returns a `Dataset` with `len() == 60_000`. Calling `get(0)` returns a sample whose image tensor has shape `[1, 28, 28]` and label is a `u8` in `0..10`.
- [ ] AC-4: `Cifar10::new(root, Split::Test, None)` returns a `Dataset` with `len() == 10_000`. Each sample image has shape `[3, 32, 32]` with values in `[0.0, 1.0]`. `Cifar100` works identically but with labels in `0..100`.
- [ ] AC-5: `ImageNet::new(root, Split::Train, Some(transform))` loads class directories from `root/train/`, infers class-to-index mapping from sorted directory names, and serves samples lazily. Passing a nonexistent root returns `Err(FerrotorchError::InvalidArgument { .. })`.
- [ ] AC-6: A `Compose` pipeline of `Resize(256, 256) -> CenterCrop(224, 224) -> ToTensor -> Normalize(IMAGENET_MEAN, IMAGENET_STD)` applied to a raw `[H, W, 3]` byte buffer produces a `Tensor<f32>` of shape `[3, 224, 224]` with per-channel normalized values. `ColorJitter`, `RandomHorizontalFlip`, `RandomResizedCrop`, `RandomAffine`, and `RandomErasing` produce valid output tensors with correct shapes. `MixUp` and `CutMix` applied to a batch of `[8, 3, 224, 224]` with labels `[8]` return tensors of the same shapes with interpolated labels.
- [ ] AC-7: All ResNet variants pass a backward test: forward a batch through the model, compute `CrossEntropyLoss`, call `backward()`, and verify all `Parameter` tensors in `model.parameters()` have non-None gradients with finite values.
- [ ] AC-8: `cargo test -p ferrotorch-vision` passes with 0 failures. Minimum 100 tests covering: every model variant's forward shape, every dataset's length and sample shape, every transform's output shape and value range, error paths for invalid inputs, and gradient flow through at least ResNet-18 and ViT-B/16.
- [ ] AC-9: `resnet50_pretrained::<f32>(1000)` loads ImageNet-trained weights from the CDN (or cache). A forward pass on a `[1, 3, 224, 224]` input produces logits of shape `[1, 1000]`. When loaded with identical weights, the output logits match PyTorch's `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)` output within a tolerance of `1e-5` per element.
- [ ] AC-10: `create_feature_extractor(resnet50_pretrained(1000)?, &["layer2", "layer4"])` returns a `FeatureExtractor` whose `forward` on a `[1, 3, 224, 224]` input returns a `HashMap` with keys `"layer2"` and `"layer4"`, where `"layer2"` has shape `[1, 512, 28, 28]` and `"layer4"` has shape `[1, 2048, 7, 7]`.
- [ ] AC-11: `models::list_models()` returns a `Vec` containing at least `"resnet18"`, `"resnet50"`, `"vgg16"`, `"efficientnet_b0"`, `"vit_b_16"`, `"convnext_tiny"`, `"swin_tiny"`, `"unet"`, and `"yolo"`. `models::get_model::<f32>("resnet50", true, 1000)` returns a model whose forward pass on `[1, 3, 224, 224]` produces `[1, 1000]`. `models::get_model::<f32>("nonexistent", false, 10)` returns `Err(FerrotorchError::InvalidArgument { .. })`.
- [ ] AC-12: `io::read_image("test.png")` returns a `RawImage` with correct width, height, and channel count. `io::write_image("out.png", &raw_image)` writes a valid PNG file that can be read back with identical pixel data. `io::read_image_as_tensor::<f32>("test.jpg")` returns a `Tensor<f32>` with shape `[3, H, W]` and values in `[0.0, 1.0]`.

## Architecture

### Crate Layout

```
ferrotorch-vision/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports, IMAGENET_MEAN / IMAGENET_STD constants
│   ├── models/
│   │   ├── mod.rs                # Re-export all model families, model registry (list_models, get_model, register_model)
│   │   ├── registry.rs           # ModelRegistry struct, global registry, dynamic model construction
│   │   ├── feature_extractor.rs  # FeatureExtractor<T> wrapper, create_feature_extractor() helper
│   │   ├── resnet.rs             # ResNet-18/34/50/101/152 (BasicBlock, Bottleneck, ResNet<T>)
│   │   ├── vgg.rs                # VGG-11/13/16/19 (VGG<T>, make_layers helper)
│   │   ├── efficientnet.rs       # EfficientNet B0-B7 (MBConv block, compound scaling)
│   │   ├── vit.rs                # ViT-B/16, ViT-L/16, ViT-H/14 (PatchEmbed, TransformerEncoder)
│   │   ├── convnext.rs           # ConvNeXt-T/S/B/L (ConvNeXtBlock, LayerNorm2d, inverted bottleneck)
│   │   ├── swin.rs               # Swin-T/S/B/L (WindowAttention, SwinBlock, PatchMerging, shifted windows)
│   │   ├── unet.rs               # U-Net (DoubleConv, Down, Up, OutConv, skip connections)
│   │   └── yolo.rs               # YOLO detection head (CSPDarknet backbone, FPN neck, detect head)
│   ├── datasets/
│   │   ├── mod.rs                # Re-export all datasets, Split enum, download utilities
│   │   ├── mnist.rs              # MNIST / FashionMNIST (IDX file parser, auto-download)
│   │   ├── cifar.rs              # CIFAR-10 / CIFAR-100 (pickle-format binary parser, auto-download)
│   │   └── imagenet.rs           # ImageNet directory loader (no download, lazy sample loading)
│   ├── io/
│   │   ├── mod.rs                # Re-export read_image, write_image, read_image_as_tensor
│   │   └── image_io.rs           # PNG/JPEG reading and writing via the `image` crate
│   └── transforms/
│       ├── mod.rs                # Re-export all transforms
│       ├── spatial.rs            # Resize, CenterCrop, RandomCrop, RandomResizedCrop, RandomHorizontalFlip, RandomAffine
│       ├── color.rs              # ColorJitter (brightness, contrast, saturation, hue)
│       ├── erasing.rs            # RandomErasing (random rectangular region erasing)
│       ├── batch.rs              # MixUp, CutMix (batch-level augmentations implementing BatchTransform)
│       └── tensor.rs             # ToTensor, Normalize
└── tests/
    ├── test_models.rs            # Forward shape checks for every model variant including ConvNeXt, Swin
    ├── test_feature_extractor.rs # Feature extraction, intermediate activation shapes
    ├── test_registry.rs          # list_models, get_model, register_model, error on unknown name
    ├── test_datasets.rs          # Length, sample shape, label range for MNIST/CIFAR/ImageNet
    ├── test_transforms.rs        # Output shape, value range, error paths for all transforms incl. MixUp/CutMix
    ├── test_image_io.rs          # read_image, write_image round-trip, format detection, error paths
    ├── test_backward.rs          # Gradient flow through ResNet-18, ViT-B/16
    └── test_pretrained.rs        # Pretrained weight loading, ResNet-50 parity with PyTorch (gated behind download-tests)
```

### Model Architecture Patterns

All models follow a consistent pattern: a struct with `Module<T>` implementation, a public constructor, a pretrained constructor, and a `forward_features` method for intermediate activation extraction.

```rust
pub struct ResNet<T: Float = f32> {
    conv1: Conv2d<T>,
    bn1: BatchNorm2d<T>,
    layers: Vec<Vec<ResNetBlock<T>>>,   // 4 layer groups, each with N blocks
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<T>,
}

impl<T: Float> Module<T> for ResNet<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, FerrotorchError> { ... }
    fn parameters(&self) -> Vec<&Parameter<T>> { ... }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> { ... }
    fn train(&mut self) { ... }
    fn eval(&mut self) { ... }
    fn named_parameters(&self) -> Vec<(&str, &Parameter<T>)> { ... }
    fn load_state_dict(&mut self, state: &StateDict<T>) -> Result<(), FerrotorchError> { ... }
    fn state_dict(&self) -> StateDict<T> { ... }
}

impl<T: Float> ResNet<T> {
    /// Returns intermediate feature maps at the named layers.
    /// Valid layer names for ResNet: "conv1", "layer1", "layer2", "layer3", "layer4".
    pub fn forward_features(
        &self,
        input: &Tensor<T>,
        layers: &[&str],
    ) -> Result<HashMap<String, Tensor<T>>, FerrotorchError> { ... }
}
```

Every model exposes a similar `forward_features` method. The valid layer names are documented per model family in their struct-level rustdoc.

ResNet uses two block types: `BasicBlock` (two 3x3 convolutions, used in ResNet-18/34) and `Bottleneck` (1x1 -> 3x3 -> 1x1, used in ResNet-50/101/152). Both include residual skip connections and batch normalization. The block type is selected at construction time based on the variant depth.

**EfficientNet** uses `MBConv` (mobile inverted bottleneck) blocks with squeeze-and-excitation. Compound scaling (width, depth, resolution multipliers) differentiates B0 through B7. The scaling coefficients are constants embedded in `efficientnet.rs`.

**ViT** splits the input image into fixed-size patches (16x16 or 14x14), projects each patch to an embedding via a single `Conv2d`, prepends a learnable `[CLS]` token, adds positional embeddings, and processes through a stack of `TransformerEncoder` layers (each using `MultiheadAttention` and a feed-forward network from ferrotorch-nn). The `[CLS]` token output feeds the classification head.

**U-Net** follows the encoder-decoder pattern: the encoder downsamples through 4 stages of `DoubleConv` + `MaxPool2d`, the decoder upsamples through 4 stages of bilinear upsampling + `DoubleConv`, with skip connections concatenating encoder features at each resolution level. The final `OutConv` is a 1x1 convolution mapping to the output class count.

**ConvNeXt** modernizes the pure convolutional design using ideas from transformers: depthwise convolution with 7x7 kernels, inverted bottleneck blocks (expand channel dimension 4x with pointwise convolution, then project back), Layer Normalization instead of Batch Normalization, and GELU activations. The four stages use `ConvNeXtBlock` with increasing channel widths (96/192/384/768 for Tiny). Downsampling between stages uses strided 2x2 convolutions with Layer Normalization. The classification head is `GlobalAvgPool -> LayerNorm -> Linear`.

**Swin Transformer** applies self-attention within local windows (7x7 by default) rather than globally, making it efficient for high-resolution inputs. `SwinBlock` alternates between regular window partitioning and shifted window partitioning (shifting by half the window size) across consecutive layers to enable cross-window connections. `PatchMerging` layers downsample spatial resolution between stages by concatenating 2x2 patches and projecting with a linear layer. The architecture has four stages with increasing channel depth (96/192/384/768 for Tiny). `WindowAttention` implements relative position bias via a learnable bias table indexed by relative position.

**YOLO** uses a CSPDarknet53 backbone for feature extraction, an FPN (Feature Pyramid Network) neck for multi-scale feature fusion, and a detection head that predicts bounding boxes, objectness scores, and class probabilities at three scales. Output format: `Tensor<T>` of shape `[batch, num_anchors * (5 + num_classes), grid_h, grid_w]` per scale, returned as a `Vec<Tensor<T>>` from a dedicated `detect` method (the `Module::forward` method returns the concatenated raw predictions for loss computation).

### Feature Extraction

The `FeatureExtractor<T>` wrapper enables returning intermediate layer activations from any model, which is essential for transfer learning, feature pyramid networks, and style transfer:

```rust
/// Wraps any model to return intermediate feature maps instead of the final output.
pub struct FeatureExtractor<T: Float> {
    model: Box<dyn Module<T>>,
    return_nodes: Vec<String>,
}

/// Convenience constructor.
pub fn create_feature_extractor<T: Float>(
    model: impl Module<T> + 'static,
    return_nodes: &[&str],
) -> Result<FeatureExtractor<T>, FerrotorchError> { ... }

impl<T: Float> FeatureExtractor<T> {
    pub fn forward(&self, input: &Tensor<T>) -> Result<HashMap<String, Tensor<T>>, FerrotorchError> { ... }
}
```

The `return_nodes` parameter specifies which named layers to capture. Invalid layer names produce `Err(FerrotorchError::InvalidArgument { .. })`.

### Model Registry

The model registry provides dynamic model construction by name, following the `torchvision.models` API:

```rust
/// Returns the names of all registered models.
pub fn list_models() -> Vec<&'static str> { ... }

/// Constructs a model by name. Returns Err for unknown names.
pub fn get_model<T: Float>(
    name: &str,
    pretrained: bool,
    num_classes: usize,
) -> Result<Box<dyn Module<T>>, FerrotorchError> { ... }

/// Registers a custom model constructor under the given name.
pub fn register_model<T: Float>(
    name: &'static str,
    constructor: fn(bool, usize) -> Result<Box<dyn Module<T>>, FerrotorchError>,
) { ... }
```

All built-in models are registered at crate initialization. The registry is stored in a global `RwLock<HashMap>` and is thread-safe.

### Image I/O

The `io` module provides convenient image reading and writing backed by the `image` crate:

```rust
/// Reads a PNG or JPEG image from disk.
pub fn read_image(path: impl AsRef<Path>) -> Result<RawImage, FerrotorchError> { ... }

/// Writes a RawImage to disk. Format inferred from extension (.png, .jpg, .jpeg).
pub fn write_image(path: impl AsRef<Path>, image: &RawImage) -> Result<(), FerrotorchError> { ... }

/// Reads an image and converts it directly to a Tensor with shape [C, H, W] in [0.0, 1.0].
pub fn read_image_as_tensor<T: Float>(path: impl AsRef<Path>) -> Result<Tensor<T>, FerrotorchError> { ... }
```

### Dataset Download and Caching

Datasets that support auto-download (MNIST, CIFAR) follow this flow:

1. Check `{root}/{dataset_name}/` for extracted files
2. If missing, check for the compressed archive in the cache directory
3. If missing, download from the canonical URL via HTTP (using `ureq`)
4. Verify the SHA-256 checksum of the downloaded archive against a hardcoded expected value
5. Extract (gunzip for MNIST IDX files, untar for CIFAR binary batches) into `{root}/{dataset_name}/`
6. Parse the binary format into in-memory vectors of `(image_bytes, label)` tuples

The `Split` enum selects train vs test data:

```rust
#[derive(Clone, Copy, Debug)]
pub enum Split {
    Train,
    Test,
}
```

ImageNet skips steps 2-5 and directly scans the directory tree at step 6. Class names are inferred from sorted subdirectory names under `{root}/{split}/`.

### Transform Pipeline

Transforms operate on a `RawImage` struct (pre-tensor) or on `Tensor<T>` (post-conversion), depending on the transform:

```rust
/// Raw image data before tensor conversion.
pub struct RawImage {
    pub data: Vec<u8>,      // RGB interleaved bytes
    pub width: usize,
    pub height: usize,
    pub channels: usize,    // 1 for grayscale, 3 for RGB
}
```

Spatial transforms (`Resize`, `CenterCrop`, `RandomCrop`, `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomAffine`) and color transforms (`ColorJitter`) operate on `RawImage`. `ToTensor` converts `RawImage` to `Tensor<T>` with shape `[C, H, W]` and values scaled to `[0.0, 1.0]`. `Normalize` operates on `Tensor<T>` with per-channel mean and standard deviation. `RandomErasing` operates on `Tensor<T>` (post-conversion), erasing a random rectangular region.

`RandomResizedCrop` is the standard training augmentation: it selects a random crop of the input with a configurable scale range (default 0.08 to 1.0 of the original area) and aspect ratio range (default 3/4 to 4/3), then resizes the crop to the target size using bilinear interpolation. `RandomAffine` applies a random affine transformation combining rotation (degrees range), translation (fraction of image size), scale (range), and shear (degrees range) with bilinear interpolation and configurable fill value. `RandomErasing` selects a random rectangle within the tensor and replaces it with random values, the image mean, or a fixed value, following the Random Erasing Data Augmentation paper.

Batch-level augmentations (`MixUp`, `CutMix`) operate on an entire batch tensor `[N, C, H, W]` and its label tensor `[N]` or `[N, num_classes]`, producing augmented tensors of the same shapes. They implement `BatchTransform` rather than `Transform` because they require pairs of samples. `MixUp` samples a mixing coefficient `lambda` from a Beta distribution and linearly interpolates both images and labels. `CutMix` samples `lambda`, computes a random bounding box whose area is `(1 - lambda)` of the image, and pastes the region from one sample onto another, adjusting labels by the area ratio.

`Resize` uses bilinear interpolation implemented directly on the byte buffer. No external image library dependency is required for the core transforms; the IDX and binary formats are parsed natively.

### Constants

```rust
/// ImageNet per-channel mean (RGB), used for normalization.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet per-channel standard deviation (RGB), used for normalization.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `FerrotorchError`, `Device` |
| `ferrotorch-nn` | workspace | `Module` trait, `Conv2d`, `Linear`, `BatchNorm2d`, `LayerNorm`, `Dropout`, `MultiheadAttention`, `Sequential`, `AdaptiveAvgPool2d`, `MaxPool2d`, loss functions |
| `ferrotorch-data` | workspace | `Dataset` trait, `Transform` trait, `BatchTransform` trait, `Compose`, `DataLoader` |
| `ferrotorch-serialize` | workspace | `StateDict`, SafeTensors loading for pretrained weights |
| `ureq` | 3.0 | HTTP client for dataset and weight downloads (blocking, minimal dependencies) |
| `sha2` | 0.10 | SHA-256 checksum verification of downloaded archives and weight files |
| `flate2` | 1.1 | Gzip decompression for MNIST IDX files |
| `image` | 0.25 | PNG/JPEG reading and writing for `io::read_image` / `io::write_image` |

### Test Strategy

1. **Forward shape tests**: For every model variant (including ConvNeXt and Swin Transformer) and every supported input size, verify the output tensor shape matches the expected dimensions. Use small batch sizes (1-2) to keep test runtime low.
2. **Dataset integration tests**: Use a `#[cfg(feature = "download-tests")]` feature gate for tests that require network access. Ungated tests use small fixture files committed to the repo (a 10-sample MNIST subset in IDX format) to verify parsing logic.
3. **Transform property tests**: Verify that `Resize` produces the requested dimensions, `CenterCrop` extracts the correct spatial region, `RandomResizedCrop` produces the target size regardless of input size, `RandomAffine` preserves spatial dimensions, `RandomErasing` only modifies the erased region, `Normalize` produces zero-mean unit-variance output when given matching mean/std, and `ColorJitter` stays within `[0, 255]` byte range. For `MixUp` and `CutMix`, verify output shapes match input shapes and that mixed labels sum to 1.0 per sample.
4. **Backward flow tests**: Forward a `[2, 3, 224, 224]` batch through ResNet-18 and ViT-B/16, compute `CrossEntropyLoss` against random labels, call `backward()`, and assert every parameter has a finite gradient.
5. **Error path tests**: Pass wrong-rank tensors to models, request out-of-range dataset indices, provide invalid transform parameters, request unknown model names from the registry — verify `FerrotorchError` variants are returned.
6. **Feature extraction tests**: Use `create_feature_extractor` and `forward_features` on ResNet-50 and ViT-B/16, verify returned feature maps have correct names and spatial dimensions at each layer.
7. **Model registry tests**: Verify `list_models()` returns all expected model names, `get_model` constructs each model successfully, `register_model` adds a custom entry that is retrievable, and unknown names return an error.
8. **Image I/O tests**: Round-trip a `RawImage` through `write_image` and `read_image`, verify pixel data is identical. Test `read_image_as_tensor` produces correct shape and value range. Test error paths for missing files and unsupported formats.
9. **Pretrained weight parity tests** (gated behind `download-tests`): Load ResNet-50 with pretrained ImageNet weights, forward a `[1, 3, 224, 224]` input, and compare output logits against a reference snapshot generated by PyTorch's `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)` with the same input, asserting element-wise agreement within `1e-5`.

## Resolved Questions

### Q1: Image decoding dependency
**Decision**: The `image` crate (0.25) is a required dependency. MNIST uses the IDX binary format (4 magic bytes + dimensions + raw bytes) and CIFAR uses its own binary batch format (label byte + 3072 pixel bytes per image), both parsed natively without `image`. However, the `io` module's `read_image` / `write_image` / `read_image_as_tensor` functions use `image` for PNG and JPEG support, and the ImageNet dataset loader uses it to decode JPEG files from disk. Making `image` a required dependency matches user expectations — a vision crate should be able to read images out of the box.

### Q2: Pretrained weight distribution
**Decision**: Pretrained weights are not bundled in the crate. The `*_pretrained` constructors and `get_model(name, pretrained=true, ...)` download SafeTensors weight files from a CDN (base URL configurable via the `FERROTORCH_WEIGHTS_URL` environment variable, defaulting to the project's release hosting). The download follows the same cache-and-verify pattern as datasets: download to `~/.ferrotorch/weights/{model_name}.safetensors`, verify SHA-256 against a hardcoded checksum table, load via ferrotorch-serialize. If the weight file already exists and passes the checksum, no download occurs. All pretrained weights are ImageNet-1K-trained. When `num_classes` differs from 1000, the classification head is re-initialized with random weights while backbone weights are loaded from the checkpoint. The weight files use SafeTensors format exclusively for safety (no arbitrary code execution) and fast memory-mapped loading.

### Q3: YOLO output format
**Decision**: YOLO's `Module::forward` returns raw predictions as a single concatenated `Tensor<T>` suitable for loss computation. A separate `Yolo::detect` method performs NMS (non-maximum suppression) post-processing and returns structured detection results. This keeps the `Module` trait implementation clean while still providing end-to-end inference via the dedicated method.

## Out of Scope
- Training loops and training utilities — users compose `DataLoader`, model `forward`, loss, and optimizer `step` themselves (or a future `ferrotorch-train` crate provides this)
- Video datasets and video models (SlowFast, I3D) — future extension
- Object detection evaluation metrics (mAP, IoU) — belongs in a metrics utility, not in the model crate
- GAN architectures (DCGAN, StyleGAN) — future extension
- Pre-trained weight hosting infrastructure — this crate defines the download protocol, not the server
- Audio or text models — separate domain crates (ferrotorch-audio, ferrotorch-text)
- RandAugment / AutoAugment / TrivialAugment policy-based augmentation — future extension
- ONNX export of vision models — handled by ferrotorch-serialize at the framework level

---
title: "Phase 5 — Vision (ferrotorch-vision)"
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-15
updated: 2026-03-15
---


## Design Specification

### Summary

A torchvision-equivalent crate providing pre-built model architectures (ResNet, VGG, EfficientNet, ViT, U-Net, YOLO), standard image datasets (MNIST, CIFAR-10/100, ImageNet loader), and image-specific transforms (Resize, CenterCrop, ColorJitter). Every model implements the `Module` trait from ferrotorch-nn, every dataset implements the `Dataset` trait from ferrotorch-data, and every transform implements the `Transform` trait from ferrotorch-data. This crate is the "batteries included" layer that lets users train and evaluate vision models out of the box without hand-rolling architectures or data pipelines.

### Requirements

- REQ-1: The crate must provide ResNet variants (18, 34, 50, 101, 152) as structs implementing `Module<T>`. Each variant must support an optional `pretrained: bool` constructor flag (loads weights via ferrotorch-serialize when true, random kaiming initialization when false) and a configurable `num_classes` parameter for the final classification head.
- REQ-2: The crate must provide VGG (11, 13, 16, 19), EfficientNet (B0 through B7), Vision Transformer (ViT-B/16, ViT-L/16, ViT-H/14), U-Net (encoder-decoder with skip connections), and YOLO (single-shot detection head with anchor-based output) — all implementing `Module<T>` with configurable `num_classes` and optional pretrained weight loading.
- REQ-3: The crate must provide dataset structs for MNIST (60K train / 10K test, 28x28 grayscale), CIFAR-10 (50K train / 10K test, 32x32 RGB, 10 classes), and CIFAR-100 (50K train / 10K test, 32x32 RGB, 100 classes), each implementing `Dataset` from ferrotorch-data. Datasets must auto-download from canonical URLs to a configurable cache directory (`~/.ferrotorch/datasets/` by default) and verify integrity via SHA-256 checksums before extraction.
- REQ-4: The crate must provide an ImageNet loader struct implementing `Dataset` that reads from a local directory in the standard `train/<class_name>/image.JPEG` layout. It must not attempt to download ImageNet (license restrictions). It must support a `Transform` pipeline applied lazily per sample.
- REQ-5: The crate must provide image-specific transforms: `Resize` (bilinear interpolation to target H x W), `CenterCrop` (extract center region), `RandomCrop` (extract random region with optional padding), `RandomHorizontalFlip` (probability-based), `ColorJitter` (random adjustments to brightness, contrast, saturation, hue), `Normalize` (per-channel mean/std), and `ToTensor` (convert raw bytes to `Tensor<T>` with shape `[C, H, W]` and values in `[0.0, 1.0]`). All transforms must implement `Transform` from ferrotorch-data.
- REQ-6: All public functions must return `Result<T, FerrotorchError>`. Invalid inputs (wrong spatial dimensions, out-of-range jitter parameters, missing dataset files, corrupted downloads) must produce descriptive errors, never panics.
- REQ-7: Every model must expose a `forward` method matching the `Module` trait signature. Models that produce non-classification outputs (U-Net segmentation maps, YOLO detection tensors) must document their output tensor shapes in their struct-level rustdoc.
- REQ-8: The crate must provide convenience constructors following the pattern `resnet18(num_classes: usize) -> Result<ResNet<T>, FerrotorchError>` and `resnet18_pretrained(num_classes: usize) -> Result<ResNet<T>, FerrotorchError>` for every model family. Pretrained weight files must use the SafeTensors format via ferrotorch-serialize.

### Acceptance Criteria

- [ ] AC-1: `resnet18::<f32>(1000)` constructs a ResNet-18 with random weights. A forward pass on a `Tensor<f32>` of shape `[1, 3, 224, 224]` returns a tensor of shape `[1, 1000]` without error. The same holds for ResNet-34/50/101/152.
- [ ] AC-2: `VGG::vgg16(1000)`, `EfficientNet::b0(1000)`, and `ViT::vit_b_16(1000)` each accept a `[1, 3, 224, 224]` input and produce a `[1, 1000]` output. `UNet::new(3, 21)` accepts `[1, 3, 256, 256]` and produces `[1, 21, 256, 256]`. `Yolo::new(80)` accepts `[1, 3, 416, 416]` and produces a detection tensor whose shape is documented and correct.
- [ ] AC-3: `Mnist::new(root, Split::Train, Some(transform))` downloads MNIST to `root` (or uses the cached copy), verifies SHA-256 checksums, and returns a `Dataset` with `len() == 60_000`. Calling `get(0)` returns a sample whose image tensor has shape `[1, 28, 28]` and label is a `u8` in `0..10`.
- [ ] AC-4: `Cifar10::new(root, Split::Test, None)` returns a `Dataset` with `len() == 10_000`. Each sample image has shape `[3, 32, 32]` with values in `[0.0, 1.0]`. `Cifar100` works identically but with labels in `0..100`.
- [ ] AC-5: `ImageNet::new(root, Split::Train, Some(transform))` loads class directories from `root/train/`, infers class-to-index mapping from sorted directory names, and serves samples lazily. Passing a nonexistent root returns `Err(FerrotorchError::InvalidArgument { .. })`.
- [ ] AC-6: A `Compose` pipeline of `Resize(256, 256) -> CenterCrop(224, 224) -> ToTensor -> Normalize(IMAGENET_MEAN, IMAGENET_STD)` applied to a raw `[H, W, 3]` byte buffer produces a `Tensor<f32>` of shape `[3, 224, 224]` with per-channel normalized values. `ColorJitter` and `RandomHorizontalFlip` produce valid output tensors with correct shapes.
- [ ] AC-7: All ResNet variants pass a backward test: forward a batch through the model, compute `CrossEntropyLoss`, call `backward()`, and verify all `Parameter` tensors in `model.parameters()` have non-None gradients with finite values.
- [ ] AC-8: `cargo test -p ferrotorch-vision` passes with 0 failures. Minimum 100 tests covering: every model variant's forward shape, every dataset's length and sample shape, every transform's output shape and value range, error paths for invalid inputs, and gradient flow through at least ResNet-18 and ViT-B/16.

### Architecture

### Crate Layout

```
ferrotorch-vision/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports, IMAGENET_MEAN / IMAGENET_STD constants
│   ├── models/
│   │   ├── mod.rs                # Re-export all model families
│   │   ├── resnet.rs             # ResNet-18/34/50/101/152 (BasicBlock, Bottleneck, ResNet<T>)
│   │   ├── vgg.rs                # VGG-11/13/16/19 (VGG<T>, make_layers helper)
│   │   ├── efficientnet.rs       # EfficientNet B0-B7 (MBConv block, compound scaling)
│   │   ├── vit.rs                # ViT-B/16, ViT-L/16, ViT-H/14 (PatchEmbed, TransformerEncoder)
│   │   ├── unet.rs               # U-Net (DoubleConv, Down, Up, OutConv, skip connections)
│   │   └── yolo.rs               # YOLO detection head (CSPDarknet backbone, FPN neck, detect head)
│   ├── datasets/
│   │   ├── mod.rs                # Re-export all datasets, Split enum, download utilities
│   │   ├── mnist.rs              # MNIST / FashionMNIST (IDX file parser, auto-download)
│   │   ├── cifar.rs              # CIFAR-10 / CIFAR-100 (pickle-format binary parser, auto-download)
│   │   └── imagenet.rs           # ImageNet directory loader (no download, lazy sample loading)
│   └── transforms/
│       ├── mod.rs                # Re-export all transforms
│       ├── spatial.rs            # Resize, CenterCrop, RandomCrop, RandomHorizontalFlip
│       ├── color.rs              # ColorJitter (brightness, contrast, saturation, hue)
│       └── tensor.rs             # ToTensor, Normalize
└── tests/
    ├── test_models.rs            # Forward shape checks for every model variant
    ├── test_datasets.rs          # Length, sample shape, label range for MNIST/CIFAR/ImageNet
    ├── test_transforms.rs        # Output shape, value range, error paths for all transforms
    └── test_backward.rs          # Gradient flow through ResNet-18, ViT-B/16
```

### Model Architecture Patterns

All models follow a consistent pattern: a struct with `Module<T>` implementation, a public constructor, and a pretrained constructor.

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
```

ResNet uses two block types: `BasicBlock` (two 3x3 convolutions, used in ResNet-18/34) and `Bottleneck` (1x1 -> 3x3 -> 1x1, used in ResNet-50/101/152). Both include residual skip connections and batch normalization. The block type is selected at construction time based on the variant depth.

**EfficientNet** uses `MBConv` (mobile inverted bottleneck) blocks with squeeze-and-excitation. Compound scaling (width, depth, resolution multipliers) differentiates B0 through B7. The scaling coefficients are constants embedded in `efficientnet.rs`.

**ViT** splits the input image into fixed-size patches (16x16 or 14x14), projects each patch to an embedding via a single `Conv2d`, prepends a learnable `[CLS]` token, adds positional embeddings, and processes through a stack of `TransformerEncoder` layers (each using `MultiheadAttention` and a feed-forward network from ferrotorch-nn). The `[CLS]` token output feeds the classification head.

**U-Net** follows the encoder-decoder pattern: the encoder downsamples through 4 stages of `DoubleConv` + `MaxPool2d`, the decoder upsamples through 4 stages of bilinear upsampling + `DoubleConv`, with skip connections concatenating encoder features at each resolution level. The final `OutConv` is a 1x1 convolution mapping to the output class count.

**YOLO** uses a CSPDarknet53 backbone for feature extraction, an FPN (Feature Pyramid Network) neck for multi-scale feature fusion, and a detection head that predicts bounding boxes, objectness scores, and class probabilities at three scales. Output format: `Tensor<T>` of shape `[batch, num_anchors * (5 + num_classes), grid_h, grid_w]` per scale, returned as a `Vec<Tensor<T>>` from a dedicated `detect` method (the `Module::forward` method returns the concatenated raw predictions for loss computation).

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

Spatial transforms (`Resize`, `CenterCrop`, `RandomCrop`, `RandomHorizontalFlip`) and color transforms (`ColorJitter`) operate on `RawImage`. `ToTensor` converts `RawImage` to `Tensor<T>` with shape `[C, H, W]` and values scaled to `[0.0, 1.0]`. `Normalize` operates on `Tensor<T>` with per-channel mean and standard deviation.

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
| `ferrotorch-data` | workspace | `Dataset` trait, `Transform` trait, `Compose`, `DataLoader` |
| `ferrotorch-serialize` | workspace | `StateDict`, SafeTensors loading for pretrained weights |
| `ureq` | 3.0 | HTTP client for dataset downloads (blocking, minimal dependencies) |
| `sha2` | 0.10 | SHA-256 checksum verification of downloaded archives |
| `flate2` | 1.1 | Gzip decompression for MNIST IDX files |

### Test Strategy

1. **Forward shape tests**: For every model variant and every supported input size, verify the output tensor shape matches the expected dimensions. Use small batch sizes (1-2) to keep test runtime low.
2. **Dataset integration tests**: Use a `#[cfg(feature = "download-tests")]` feature gate for tests that require network access. Ungated tests use small fixture files committed to the repo (a 10-sample MNIST subset in IDX format) to verify parsing logic.
3. **Transform property tests**: Verify that `Resize` produces the requested dimensions, `CenterCrop` extracts the correct spatial region, `Normalize` produces zero-mean unit-variance output when given matching mean/std, and `ColorJitter` stays within `[0, 255]` byte range.
4. **Backward flow tests**: Forward a `[2, 3, 224, 224]` batch through ResNet-18 and ViT-B/16, compute `CrossEntropyLoss` against random labels, call `backward()`, and assert every parameter has a finite gradient.
5. **Error path tests**: Pass wrong-rank tensors to models, request out-of-range dataset indices, provide invalid transform parameters — verify `FerrotorchError` variants are returned.

### Out of Scope

- Training loops and training utilities — users compose `DataLoader`, model `forward`, loss, and optimizer `step` themselves (or a future `ferrotorch-train` crate provides this)
- Video datasets and video models (SlowFast, I3D) — future extension
- Object detection evaluation metrics (mAP, IoU) — belongs in a metrics utility, not in the model crate
- GAN architectures (DCGAN, StyleGAN) — future extension
- Pre-trained weight hosting infrastructure — this crate defines the download protocol, not the server
- Audio or text models — separate domain crates (ferrotorch-audio, ferrotorch-text)
- Image augmentation beyond the listed transforms (MixUp, CutMix, RandAugment) — future extension
- ONNX export of vision models — handled by ferrotorch-serialize at the framework level

### resolved questions

### Q1: Image decoding dependency
**Decision**: No external image decoding library. MNIST uses the IDX binary format (4 magic bytes + dimensions + raw bytes). CIFAR uses its own binary batch format (label byte + 3072 pixel bytes per image). ImageNet requires pre-decoded images — the user is responsible for decoding JPEG/PNG to raw bytes before passing to the dataset (or using a `DecodeImage` transform backed by the `image` crate, which is an optional dependency behind a feature flag `image-decode`). This keeps the core dependency tree minimal.

### Q2: Pretrained weight distribution
**Decision**: Pretrained weights are not bundled in the crate. The `*_pretrained` constructors download SafeTensors weight files from a configurable URL (defaulting to a GitHub release or similar hosting). The download follows the same cache-and-verify pattern as datasets: download to `~/.ferrotorch/weights/{model_name}.safetensors`, verify SHA-256, load via ferrotorch-serialize. If the weight file already exists and passes the checksum, no download occurs.

### Q3: YOLO output format
**Decision**: YOLO's `Module::forward` returns raw predictions as a single concatenated `Tensor<T>` suitable for loss computation. A separate `Yolo::detect` method performs NMS (non-maximum suppression) post-processing and returns structured detection results. This keeps the `Module` trait implementation clean while still providing end-to-end inference via the dedicated method.


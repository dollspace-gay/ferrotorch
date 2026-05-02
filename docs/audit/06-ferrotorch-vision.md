# Audit: `ferrotorch-vision` vs `torchvision`

torchvision is a separate repo (not in `~/pytorch`) — comparison is from
public knowledge of its API, not direct clone diff.

## Scope

- `torchvision.models` (60+ classification + detection + segmentation +
  video models)
- `torchvision.datasets` (~50 datasets)
- `torchvision.transforms` (v1) + `torchvision.transforms.v2` (~40 ops)
- `torchvision.io` (image/video read/write)
- `torchvision.ops` (NMS, RoIAlign, RoIPool, FPN, etc.)
- `torchvision.utils` (`make_grid`, `save_image`, `draw_bounding_boxes`)
- `torchvision.tv_tensors` (TVTensor wrappers)
- `torchvision.prototype` (experimental)

## Models

ferrotorch-vision (13 families):
- **Classification**: ConvNeXt, DenseNet, EfficientNet, Inception, MobileNet,
  ResNet, Swin, VGG, ViT
- **Segmentation**: U-Net
- **Detection**: YOLO
- (+ `FeatureExtractor`, `ModelConstructor`, `ModelRegistry`,
  `create_feature_extractor`, `get_model`, `list_models`, `register_model`)

torchvision (60+ classification, plus detection, segmentation, video).

| Category | ferrotorch | torchvision | Coverage |
|---|---|---|---|
| Classic CNN | VGG, ResNet, DenseNet, Inception, MobileNet, EfficientNet | + AlexNet, GoogLeNet, ResNeXt, Wide ResNet, SqueezeNet, ShuffleNet, MNASNet, RegNet | ~70% of popular |
| Modern arch | ConvNeXt, ViT, Swin | + MaxViT, MViT, Swin v2, MobileViT | partial |
| Detection | YOLO | FasterRCNN, MaskRCNN, KeypointRCNN, RetinaNet, FCOS, SSD, SSDLite | YOLO present (which torchvision doesn't have!), but no R-CNN family |
| Segmentation | U-Net | FCN, DeepLabV3, LRASPP | torchvision-style models missing (#457) |
| Video | none | R3D, MC3, R2Plus1D, S3D, MViT, Swin3D | gap |
| Optical flow | none | RAFT | gap |

**Coverage estimate: ~25% by model count, but ~70-80% of the
classification models people actually use.**

## Datasets

ferrotorch-vision (3): MNIST, CIFAR10, CIFAR100.

torchvision (~50): MNIST/EMNIST/FashionMNIST/KMNIST/QMNIST, CIFAR10/100,
ImageNet, Imagenette, Places365, COCO (Detection + Captions), Cityscapes,
VOC, SBD, STL10, SVHN, USPS, Kitti, Omniglot, LSUN, CelebA, Caltech-101/256,
Flickr8k/30k, LFW, Country211, CLEVR, FGVCAircraft, OxfordIIITPet, EuroSAT,
Food101, ImageFolder, DatasetFolder, video datasets, etc.

| Dataset | ferrotorch | torchvision |
|---|---|---|
| MNIST family | ✅ MNIST | + EMNIST, FashionMNIST, KMNIST, QMNIST |
| CIFAR | ✅ CIFAR10, CIFAR100 | ✅ |
| ImageNet | none | ✅ |
| COCO | none | ✅ Detection + Captions |
| ImageFolder/DatasetFolder | none | ✅ — **most-used helper** |
| All others | none | ✅ |

**ImageFolder is the single biggest gap** — it's the dataset everyone uses
for custom data ("point at a folder, get a Dataset"). Should be a quick
add.

## Transforms

ferrotorch-vision (~17): `CenterCrop`, `ColorJitter`, `Compose`,
`ElasticTransform`, `GaussianNoise`, `RandomApply`, `RandomChoice`,
`RandomCrop`, `RandomGaussianBlur`, `RandomHorizontalFlip`,
`RandomResizedCrop`, `RandomRotation`, `RandomVerticalFlip`, `Resize`,
`ToTensor`, `TrivialAugmentWide`, `VisionNormalize`.

torchvision.transforms.v2 (~40): + `Pad`, `RandomAffine`,
`RandomPerspective`, `RandomErasing`, `RandomGrayscale`,
`RandomInvert`, `RandomPosterize`, `RandomSolarize`,
`RandomAdjustSharpness`, `RandomAutocontrast`, `RandomEqualize`,
`RandAugment`, `AutoAugment`, `AugMix`, `Lambda`, `LinearTransformation`,
`Grayscale`, `RandomOrder`, `FiveCrop`, `TenCrop`,
`ConvertImageDtype`, `RandomChannelPermutation`,
detection-specific (`ClampBoundingBoxes`, `ConvertBoundingBoxFormat`,
`SanitizeBoundingBoxes`, `ConvertPointsFormat`).

**Coverage ~40%.** Big absences: `Pad`, `RandomAffine/Perspective/Erasing`,
`RandAugment`/`AutoAugment`/`AugMix`, `Grayscale`, all detection-aware
transforms (bbox/point/mask co-transforms — needed for detection
training).

## I/O

ferrotorch-vision (`io.rs`): `RawImage`, `read_image`,
`read_image_as_tensor`, `read_image_rgba`, `raw_image_to_tensor`,
`tensor_to_raw_image`, `write_image`, `write_tensor_as_image`.

torchvision.io: `read_image`, `write_jpeg`, `write_png`, `decode_image`,
`encode_jpeg`, `encode_png`, `read_video`, `write_video`,
`read_video_timestamps`, `read_file`.

| Capability | ferrotorch | torchvision |
|---|---|---|
| Image read | ✅ multiple variants | ✅ |
| Image write | ✅ | ✅ JPEG + PNG |
| Video read | ❌ | ✅ |
| Video write | ❌ | ✅ |
| Format-specific encode/decode | ❌ | ✅ |

**Video I/O is a complete gap.** Required for any video model port.

## Ops

ferrotorch-vision: **none in `lib.rs`** (no `ferrotorch_vision::ops`).

torchvision.ops: `nms`, `batched_nms`, `box_iou`, `complete_box_iou`,
`distance_box_iou`, `generalized_box_iou`, `box_area`,
`clip_boxes_to_image`, `remove_small_boxes`, `box_convert`,
`MultiScaleRoIAlign`, `RoIAlign`, `roi_align`, `RoIPool`, `roi_pool`,
`PSRoIAlign`, `PSRoIPool`, `FrozenBatchNorm2d`, `MLP`, `FeaturePyramidNetwork`,
`StochasticDepth`, `DropBlock2d`, `DeformConv2d`, `Permute`,
`SqueezeExcitation`, `Conv2dNormActivation`, `Conv3dNormActivation`,
`focal_loss`, `sigmoid_focal_loss`, `generalized_box_iou_loss`,
`distance_box_iou_loss`, `complete_box_iou_loss`.

**Complete absence of `ferrotorch-vision::ops` is the second-biggest gap**
(after datasets). NMS, RoIAlign, focal loss, FPN are required for any
detection or segmentation pipeline.

## Utils

ferrotorch-vision: none.

torchvision.utils: `make_grid` (image grid), `save_image`,
`draw_bounding_boxes`, `draw_segmentation_masks`, `draw_keypoints`,
`flow_to_image`.

**Complete gap.** All of these are 1-2 day adds.

## TV tensors / prototype

torchvision recently added `TVTensor` wrappers (`Image`, `Mask`,
`BoundingBoxes`, `Video`) so transforms can dispatch on type and
co-transform image+bbox+mask in one call.

ferrotorch-vision has no equivalent. This is a design choice — co-transforms
on detection/segmentation labels are tricky in Rust without a similar
wrapper type.

## Recommendations (priority-ordered)

1. **Add `ImageFolder` and `DatasetFolder`** — single biggest practical
   omission. ~200 LOC each.
2. **Add `ferrotorch-vision::ops`**: `nms`, `box_iou`, `box_convert`,
   `roi_align`, `roi_pool`, `focal_loss`, `sigmoid_focal_loss`. Required
   for detection.
3. **Add video I/O** if video models are in scope; otherwise document as
   non-goal.
4. **Add transform tail**: `Pad`, `RandomAffine`, `RandomPerspective`,
   `RandomErasing`, `Grayscale`, `RandAugment`/`AutoAugment`/`AugMix`.
5. **Add `make_grid`, `save_image`, `draw_bounding_boxes`** in a new
   `utils.rs`.
6. **Add ImageNet-style large datasets**: ImageNet (or rather, an
   `ImageNet`-shaped helper), Places365, Caltech if/when needed.
7. **Add detection model family** (FasterRCNN/RetinaNet/SSD) — large work,
   tracked separately as #456.
8. **Add segmentation model family** (FCN/DeepLabV3/LRASPP) — tracked as
   #457.
9. **Decide on `TVTensor`-like wrappers** for co-transforming detection
   labels. Probably worth adopting if detection is a goal.

## Status

**Coverage: ~25-30% by surface area, ~50% by practical
ML-engineer-typical-use.** ResNet/ViT/EfficientNet families work. Detection
and video are the biggest absences.

**Do not split.** torchvision is a single package; mirror that.

## Issues already tracked
- #456 — vision detection models (Faster R-CNN, etc.)
- #457 — vision segmentation models (DeepLab, FCN)

# YOLO-CIR same-protocol reimplementation

This experiment is an **independent, paper-guided reimplementation**. It is not
the authors' source code and must not be described as an official reproduction.

## Fixed public design evidence

- YOLOv5m is the stated starting detector.
- The backbone is an improved ConvNeXt feature extractor.
- Coordinate attention is inserted in ConvNeXt blocks.
- Spatial attention is applied to shallow, high-resolution features.
- Channel attention is applied to deep, semantic features.
- Split attention is used in the neck.

Source: Zhou et al., *Infrared Physics & Technology* 131 (2023), 104703,
<https://doi.org/10.1016/j.infrared.2023.104703>.

## Explicit reconstruction choices

- ConvNeXt stage depths: `[3, 3, 9, 3]`.
- Stage channels: `[80, 160, 320, 640]`.
- Three detection scales with strides 8, 16, and 32.
- Ultralytics C3 neck blocks and anchor-free Detect head.
- Two-radix split attention after each neck fusion block.
- ConvNeXt-Nano ImageNet initialization for compatible backbone tensors.
- Coordinate attention, neck and detection layers start from random weights.

These choices are recorded in `reimplementation_audit.json` and must be
reported as implementation assumptions because the paper and available public
materials do not uniquely define the layer-level graph.

## Same-protocol settings

The Night-Drone Vehicle data YAML, train/validation split, 300 epochs, batch 32,
640-pixel input, SGD settings, seed, augmentation, and evaluator match the
AG-YOLO experiment. Initialization is the one material difference: the custom
ConvNeXt backbone is incompatible with `yolo11n.pt`, so an architecture-matched
ImageNet backbone checkpoint is used and disclosed.

# EMSFFN Architecture Notes

The Enhanced Multi-Scale Feature Fusion Network (EMSFFN) is built on YOLOv8.

Key modifications:

1. Keep the original YOLOv8 backbone feature extraction network.
2. Add TBAM at the end of the backbone network.
3. In the FPN path:
   - Use C3 (80×80), C2 (160×160), C1 (320×320).
   - M3 (80×80) is upsampled and concatenated with C2, then processed by C2f to generate M2 (160×160).
   - M2 is upsampled and concatenated with C1, then processed by C2f to generate M1 (320×320).
4. In the PAN path:
   - P1 (320×320) is downsampled and merged with M2 to generate P2 (160×160).
   - A new prediction head Head4 is added after P2 for small target detection.
   - TBAM is inserted before each prediction head.
5. Four prediction heads are used:
   - Head1: 20×20
   - Head2: 40×40
   - Head3: 80×80
   - Head4: 160×160

This design enhances small defect target perception while maintaining a relatively low computational cost.

> Note: The YAML file `yolov8_tbam_emsffn.yaml` is a reference template.  
> For exact reproduction, use the original training code. The layer indices and channel numbers must match the Ultralytics version used in the experiments.

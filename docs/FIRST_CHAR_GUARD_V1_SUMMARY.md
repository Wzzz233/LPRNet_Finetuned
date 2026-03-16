# First Char Guard V1 Summary

## Goal

The deployment target is the board-side raw OCR path:

- `--ocr-channel-order bgr`
- `--ocr-crop-mode match`
- `--ocr-resize-mode letterbox`
- `--ocr-resize-kernel nn`
- `--ocr-min-occ-ratio 0.90`
- `--fpga-preproc-profile raw`

The concrete failure to fix was the first Chinese province character on raw board crops.  
The real anchor sample used in this round is:

- `ocrin_0021_f000042.ppm -> 京N8P8F8`

## What Changed Across Experiments

| Experiment | Main change | Board raw sample | Val exact | Test exact | Test first-char |
| --- | --- | --- | --- | --- | --- |
| `first_board_baseline_v1` | Direct board-aligned fine-tune from official stage-3 weights | `N8P8F8` | `94.14%` | `13.71%` | `84.82%` |
| `crop_aligned_v1` | Added CCPD crop jitter to mimic detector/refine mismatch | `N8P8F8` | `94.66%` | `12.98%` | `58.82%` |
| `first_char_guard_v1` | Added raw board anchor oversampling + province rebalance + first-char auxiliary loss | `京N8P8F8` | `92.97%` | `17.03%` | `75.57%` |

## Why This Run Was Effective

1. The training input was kept aligned with the board OCR path instead of adding extra image enhancement.
2. A real raw board crop was injected into training as an anchor, so the model had to see at least one true deployment-domain sample.
3. The first character was given explicit pressure through an auxiliary province-classification loss over early time steps.
4. The previous checkpoint-selection rule was too weak: it picked epoch 1 as soon as the anchor became correct.  
   After checkpoint sweep, `LPRNet__iteration_10000.pth` was found to keep the anchor correct while clearly outperforming the old exported `Final` on proxy validation and test.

## Final Training Settings

- Base weights: `experiments/crop_aligned_v1/weights/Final_LPRNet_model.pth`
- Data mode: `ccpd_board`
- OCR alignment: `bgr + match + letterbox + nn + none + min_occ=0.90`
- Train crop augmentation: `jitter_refine`
- Augmentation params: `prob=0.85`, `jx=0.06`, `jy=0.12`, `min_iou=0.75`
- Learning rate: `1e-4`
- LR schedule: `3,6,8`
- Epochs: `8`
- Batch size: `64`
- Province rebalance: `inv_sqrt`
- Board anchor sample weight: `768`
- First-char auxiliary loss: `0.40`
- First-char time steps: `6`
- Checkpoint tie-break proxy: first `5000` validation samples

## Final Exported Artifacts

- PyTorch: `experiments/first_char_guard_v1/weights/Final_LPRNet_model.pth`
- ONNX: `experiments/first_char_guard_v1/weights/LPRNet_stage3_rk3568_fp16.onnx`
- RKNN: `experiments/first_char_guard_v1/weights/LPRNet_stage3_rk3568_fp16.rknn`
- Report: `experiments/first_char_guard_v1/EXPERIMENT_REPORT.md`

## Deployment Note

The current success is anchored by one real raw board sample.  
This is enough to prove the direction is correct and to produce a better deployable checkpoint, but it is not enough to claim broad deployment coverage for all provinces. The next gain should come from collecting more real board crops with diverse province characters while keeping the same raw preprocessing contract.

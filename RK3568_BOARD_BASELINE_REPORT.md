# RK3568 Board-Aligned LPRNet Baseline Report

## Objective

This experiment targets a practical deployment issue on the RK3568 board OCR path:

- the official FP16 LPRNet could often read the trailing alphanumeric characters,
- but the first Chinese province character was unstable or incorrect after board-side preprocessing.

The goal of this run was not to chase the best offline CCPD score. It was to produce a first board-aligned fine-tune that:

- keeps preprocessing aligned with the ARM + NPU inference path,
- uses a strict train/val/test split,
- finishes cleanly in one pass,
- and gives a clear measurable lift on the province character and full plate decoding.

## Why This Training Is Valid

This fine-tune is valid for board deployment because the training input distribution was deliberately moved closer to the board OCR input rather than to the raw CCPD crops.

The key change is in [load_data.py](/home/wzzz/LPRNet/load_data.py#L305), where `CCPDBoardDataLoader` reconstructs the OCR crop from the full CCPD image using the bounding box embedded in the CCPD filename and then applies board-aligned preprocessing:

- channel order: `bgr`
- crop mode: `match`
- resize mode: `letterbox`
- resize kernel: `nn`
- extra preprocessing: `none`
- min occupancy ratio: `0.90`

This matters because the board does not feed the model idealized plate crops. It feeds a resized OCR crop with its own aspect-ratio behavior and padding pattern. If training stays on a different preprocessing path, the model learns the wrong input distribution and the first character, which is usually the most sensitive to left-edge crop geometry, degrades first.

In other words, this experiment is useful even though the final strict test score is not yet deployment-grade, because it proves the domain-alignment hypothesis:

- before fine-tuning under board-aligned preprocessing, the first-character accuracy was extremely poor,
- after fine-tuning, that first-character accuracy rose sharply,
- and the gains are not limited to the validation loop inside training; they remain visible on the held-out strict test split as well.

## Dataset And Split Discipline

The label files were generated from CCPD official split files with [prepare_ccpd_splits.py](/home/wzzz/LPRNet/prepare_ccpd_splits.py#L22).

The split statistics were verified with [analyze_ccpd_splits.py](/home/wzzz/LPRNet/analyze_ccpd_splits.py#L36) and stored in [split_stats.json](/home/wzzz/LPRNet/experiments/first_board_baseline_v1/split_stats.json).

Split sizes:

- train: `100000`
- val: `99996`
- test: `141982`

Overlap check:

- train/val overlap: `0`
- train/test overlap: `0`
- val/test overlap: `0`

This means the result is not inflated by path leakage across splits.

## Training Configuration

The first formal baseline run is scripted in [run_first_board_experiment.sh](/home/wzzz/LPRNet/run_first_board_experiment.sh#L1).

Model and initialization:

- architecture: `LPRNet`
- starting weights: `weights_red_stage3/Final_LPRNet_model.pth`
- training mode: fine-tune, loading only matching parameters

Optimization:

- optimizer: `RMSprop`
- learning rate: `0.001`
- momentum: `0.9`
- weight decay: `2e-5`
- LR schedule epochs: `[4, 8, 12, 14, 16]`
- max epoch: `15`

Batching:

- train batch size: `64`
- eval batch size: `120`
- num workers: `4`
- device: `cuda`

Board-aligned OCR preprocessing:

- `--data_mode ccpd_board`
- `--ocr_channel_order bgr`
- `--ocr_crop_mode match`
- `--ocr_resize_mode letterbox`
- `--ocr_resize_kernel nn`
- `--ocr_preproc none`
- `--ocr_min_occ_ratio 0.90`

Why these values were chosen:

- `match` keeps the crop closest to the CCPD plate bbox and mirrors the board OCR assumption that the detector already localized the plate.
- `letterbox` preserves aspect ratio and reproduces the width occupancy effect seen in embedded OCR pipelines.
- `nn` was kept because the board-side OCR path is closer to nearest-neighbor style resizing than to smoother desktop preprocessing.
- `none` avoids inventing an extra preprocessing stage that does not exist on the current board path.
- `0.90` for occupancy is a conservative recrop threshold to avoid overly narrow character packing after letterboxing.
- `15` epochs was enough to complete a first stable run without turning the first experiment into an open-ended search.

## Training Outcome

Training converged cleanly. Average training loss dropped from roughly `0.4116` in epoch 1 to `0.0312` by epoch 15.

Artifacts are stored under [experiments/first_board_baseline_v1](/home/wzzz/LPRNet/experiments/first_board_baseline_v1):

- final weight: [Final_LPRNet_model.pth](/home/wzzz/LPRNet/experiments/first_board_baseline_v1/weights/Final_LPRNet_model.pth)
- train log: [train.log](/home/wzzz/LPRNet/experiments/first_board_baseline_v1/train.log)
- validation metrics: [val_metrics.json](/home/wzzz/LPRNet/experiments/first_board_baseline_v1/val_metrics.json)
- test metrics: [test_metrics.json](/home/wzzz/LPRNet/experiments/first_board_baseline_v1/test_metrics.json)

## Accuracy Summary

Pretrained board-aligned baseline on validation:

- exact plate acc: `0.004100`
- char acc: `0.113400`
- province first-char acc: `0.066713`

Fine-tuned model on validation:

- exact plate acc: `0.941408`
- length-correct acc: `0.976279`
- char acc: `0.985512`
- province first-char acc: `0.989220`
- pos2 alpha acc: `0.997610`
- pos3+ alnum acc: `0.982351`

Pretrained board-aligned baseline on strict test:

- exact plate acc: `0.002698`
- char acc: `0.086742`
- province first-char acc: `0.059684`

Fine-tuned model on strict test:

- exact plate acc: `0.137067`
- length-correct acc: `0.297763`
- char acc: `0.509963`
- province first-char acc: `0.848178`
- pos2 alpha acc: `0.771260`
- pos3+ alnum acc: `0.390061`

## Interpretation

This run is effective because it solves the primary board complaint:

- first-character recognition improved from `5.97%` to `84.82%` on the strict test split,
- and from `6.67%` to `98.92%` on validation under the exact same board-aligned preprocessing.

That is too large to be explained by noise. The model clearly learned the board-preprocessed distribution much better than the original pretrained weight.

At the same time, the held-out test score shows the baseline is not yet the final deployment model:

- the strict test set contains more difficult CCPD domains such as blur, rotation, tilt, challenge, and dark/dirty samples,
- the province distribution is heavily biased toward `皖`,
- and the post-second-character positions still lose substantial accuracy under harder domains.

So the correct conclusion is:

- this baseline successfully validates the training direction,
- it already provides a likely practical uplift on the board for the first Chinese character,
- but a second round is still needed to improve robustness on difficult domains and non-dominant provinces.

## ONNX And RKNN Export

The export chain for this run is:

1. export PyTorch weights to RKNN-friendly ONNX with [export_onnx_rknn_compatible.py](/home/wzzz/LPRNet/export_onnx_rknn_compatible.py#L1)
2. convert ONNX to RKNN with [custom_rknn_convert.py](/home/wzzz/LPRNet/custom_rknn_convert.py#L1)
3. verify output consistency with [verify_export_consistency.py](/home/wzzz/LPRNet/verify_export_consistency.py#L1)

Generated deployment artifacts:

- ONNX: [LPRNet_stage3_rk3568_fp16.onnx](/home/wzzz/LPRNet/experiments/first_board_baseline_v1/weights/LPRNet_stage3_rk3568_fp16.onnx)
- RKNN: [LPRNet_stage3_rk3568_fp16.rknn](/home/wzzz/LPRNet/experiments/first_board_baseline_v1/weights/LPRNet_stage3_rk3568_fp16.rknn)

The RKNN conversion was done for `rk3568` because the existing board-side docs and current OCR deployment references in this repo all point to that platform.

The converter explicitly uses:

- `target_platform='rk3568'`
- `do_quantization=False`
- `float_dtype='float16'`
- mean/std equivalent to the training normalization: `(x - 127.5) / 128.0`

Why this is the correct FP16 path:

- in RKNN Toolkit2, `config()` documents `float_dtype='float16'` as the non-quantized datatype default,
- and `build(do_quantization=False)` keeps the model in the non-quantized floating path rather than int8/u8 calibration.

This is exactly what we want for “keep FP16 precision”.

## Reproducible Export Command

Use [export_first_board_rknn.sh](/home/wzzz/LPRNet/export_first_board_rknn.sh#L1):

```bash
cd /home/wzzz/LPRNet
./export_first_board_rknn.sh first_board_baseline_v1
```

By default it uses:

- training env: `/home/wzzz/LPRNet/.conda`
- RKNN env: `/root/miniconda3/envs/rknn_env`

You can override the RKNN environment path like this:

```bash
RKNN_ENV_PREFIX=/path/to/your/rknn_env ./export_first_board_rknn.sh first_board_baseline_v1
```

## Next Recommended Experiment

The next iteration should keep the same board-aligned preprocessing and strict split discipline, but change the training mix:

- oversample non-`皖` provinces,
- upweight hard domains (`blur`, `rotate`, `tilt`, `challenge`, `db`, `fn`),
- keep the same export path so board-side behavior stays comparable,
- evaluate the second run against this baseline, not against raw pretrained weights only.

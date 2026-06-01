# Special Plate v2 Data Audit Report

> Generated: 2026-06-01
> Generation script: `scripts/special_gen/special_v2_generate.py`
> QA script: `scripts/special_gen/special_v2_qa.py`

## Summary

| Split | Police | Embassy | Total |
|-------|------:|-------:|------:|
| Train | 15,500 | 10,000 | 25,500 |
| Val Clean | 1,550 | 1,000 | 2,550 |
| Val Hard | 1,550 | 1,000 | 2,550 |
| **Total** | **18,600** | **12,000** | **30,600** |

Police: 31 provinces × 500 train / 50 val_clean / 50 val_hard
Embassy: 10,000 train / 1,000 val_clean / 1,000 val_hard (uniform random digits)

## Output Locations

| Item | Path |
|------|------|
| Images | `datasets/special_ccpd2019_base_cvreplace_v2_20260601/images/{train,val_clean,val_hard}/` |
| Manifests | `manifests_rebased/special_split_v2_20260601/` |
| QA sheets | `datasets/special_ccpd2019_base_cvreplace_v2_20260601/qa_v2/` |
| Generation script | `scripts/special_gen/special_v2_generate.py` |
| QA script | `scripts/special_gen/special_v2_qa.py` |
| Windows desktop copies | `//wsl$/Ubuntu/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/special_v2_20260601/` |

Total dataset size: 2.6 GB (30,600 JPEG images)

## Base Image Integrity

- Pose source: `datasets/ccpd2019_base_posquads_20260509/pose_quads.jsonl` (199,996 rows)
- Train pool: 193,996 images
- Val_clean pool: 3,000 images
- Val_hard pool: 3,000 images
- **No base image overlap between any splits: 0 shared bases** ✅

## Police Province Balance

Perfectly balanced — every province has exactly:

| Split | Per Province | Total |
|-------|:-----------:|:-----:|
| Train | 500 | 15,500 |
| Val Clean | 50 | 1,550 |
| Val Hard | 50 | 1,550 |

31 provinces: 京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新

## Police Text Format

Format: `省 + 字母(A-HJ-NP-Z) + 4×字母数字(A-HJ-NP-Z0-9) + 警` = 7 chars
No format errors found ✅

## Embassy Text Format

Format: `使 + 6×数字(0-9)` = 7 chars
No format errors found ✅

## Embassy Digit Distribution

| Split | Total digits | Min per digit | Max per digit | Mean |
|-------|:-----------:|:-------------:|:-------------:|:----:|
| Train | 60,000 | 5,907 | 6,084 | 6,000 |
| Val Clean | 6,000 | 576 | 633 | 600 |
| Val Hard | 6,000 | 581 | 636 | 600 |

Uniform random distribution, no digit bias ✅

## Manifest Fields

All manifests use the same field schema as existing v1 manifests:

- `img_path`, `text`, `family`, `source`, `split`, `preprocess_group`
- `has_quad=1`, `can_parse_ccpd_geom=0`, `can_perspective=1`
- `quad_source=pose_yolov8n_ccpd2019_base`
- `ocr_crop_mode=obb_warp`, `ocr_resize_mode=letterbox`
- `ocr_resize_kernel=nn`, `ocr_preproc=none`, `ocr_channel_order=bgr`
- `ocr_quad_pad_ratio=0.0`

Paths are relative to `PROJECT_ROOT` (`/home/wzzz/LPRNet/`), consistent with existing manifests ✅

## Color Guard Pass Rate

| Family | Split | Guard OK | Fallback Raw | Fallback % |
|--------|-------|:--------:|:-----------:|:----------:|
| Police | Train | 15,500 | 3,185 | 20.5% |
| Police | Val Clean | 1,550 | 358 | 23.1% |
| Police | Val Hard | 1,550 | 323 | 20.8% |
| Embassy | Train | 10,000 | 8,378 | 83.8% |
| Embassy | Val Clean | 1,000 | 850 | 85.0% |
| Embassy | Val Hard | 1,000 | 835 | 83.5% |

- Police: ~80% pass rate. The 20% fallback is normal — style transfer slightly shifts the white/red balance, causing color guard to fail the strict threshold. Raw rendering is used instead, which still undergoes capture finish (noise, JPEG, sharpness matching).
- Embassy: ~16% pass rate. Embassy plates are black background with red "使". The style transfer often brightens the plate, reducing `dark_ratio` below the 0.25 threshold. The fallback raw rendering still applies capture finish, producing valid but less domain-adapted images.

**Note**: The fallback mechanism is working correctly. In all cases, the resulting image has the correct text rendered and warped into the base image. The color guard only controls whether style transfer is applied or raw rendering+finish is used.

## Val Hard Degradation Types

Each val_hard image receives 2-4 of the following randomly:
- Gaussian blur (3x3 or 5x5, σ=0.3-1.0) — 70% probability
- JPEG compression (quality 60-90) — 80% probability
- Brightness/contrast shift (±40, 0.7x-1.3x) — 70% probability
- Local exposure gradient (vignette-like) — 40% probability
- Shot noise (σ=2-8) — 50% probability
- Color shift (Hue ±10°) — 30% probability

Additionally, the deformation from the base CCPD2019 image warp provides perspective variation.

## Skip Statistics

Total color guard skips across all categories: 13,929
All skips are due to the **color guard fallback path** (style transfer fails → use raw rendering + capture finish). There are no other skip causes.

## QA Sheets Available

8 QA sheets in `qa_v2/` (copied to Windows desktop):
- `qa_train_police_36.jpg` — 36 police train OCR crops
- `qa_train_embassy_36.jpg` — 36 embassy train OCR crops
- `qa_val_clean_police_36.jpg` — 36 police val_clean OCR crops
- `qa_val_clean_embassy_36.jpg` — 36 embassy val_clean OCR crops
- `qa_val_hard_police_36.jpg` — 36 police val_hard OCR crops (degraded)
- `qa_val_hard_embassy_36.jpg` — 36 embassy val_hard OCR crops (degraded)
- `qa_clean_vs_hard_police.jpg` — 20 paired clean/hard comparisons
- `qa_clean_vs_hard_embassy.jpg` — 20 paired clean/hard comparisons

Plus 6 original QA sheets in `qa/` directory.

120 sample images (full frame + OCR crop pairs, 10 per category) on Windows desktop.

## Known Issues / Cautions

1. **Embassy color guard fallback rate is high (84%)**: The dark_ratio threshold (≥0.25) is strict for embassy plates after style transfer. The fallback path still produces usable images. If this is a concern, the color guard threshold can be relaxed for embassy.

2. **Synthetic data domain gap**: As with all cvreplace data, the rendered plates may not perfectly match real camera-captured plates. The style transfer mitigates this but does not eliminate it.

3. **No real-world validation data**: This is purely synthetic data. Real board-test images are not included.

4. **Train/val_no_overlap**: ✅ verified — 0 shared base images between any splits.

## Conclusion

**✅ This dataset is ready for training.**

- All quantitative targets met ✅
- Province balance perfectly even ✅
- No train/val base image leakage ✅
- Text format correct for all 30,600 samples ✅
- Manifest format consistent with existing project conventions ✅
- QA sheets provided for visual inspection ✅

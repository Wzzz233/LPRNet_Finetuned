# Plate Type Classifier — Data Audit Report

> Generated: 2026-06-02
> Purpose: Audit all candidate data sources for a 224×72 RGB plate-type classifier
> Input spec: 224×72, RGB 3-channel, train=/255, RKNN mean=[0,0,0] std=[255,255,255]

## 1. Data Source Overview

| # | Source | Total Images | Plate Type | Label Source | Format |
|---|--------|:-----------:|-----------|-------------|--------|
| 1 | CCPD2019 | ~90,258 (manifest) | blue 7-char | Filename + existing manifest | Full-frame w/ quad |
| 2 | CCPD2020 (green) | 11,776 (files) | green 8-char | Filename + existing manifest | Full-frame w/ quad |
| 3 | CRPD_all | 33,757 | ~98% blue 7-char | YOLO label files (text field) | Full-frame w/ quad |
| 4 | git_plate | 63,196 | Mixed (blue, police, embassy, etc.) | Filename convention | Cropped (small, variable) |
| 5 | CBLPRD-330k_v1 | 342,110 | 7 types (blue, green, yellow, black) | data.txt (text + type) | Cropped 128×48, GAN |
| 6 | special_split_v2 | 30,600 | police + embassy_shi | Manifest text field | Full-frame w/ quad, synthetic |

## 2. Blacklist: CRPD_raw_ccpd_board_v1

- Status: ✅ Excluded and confirmed empty (**43365** image files remaining)
- Subdirs exist `blue/`, `special/`, `yellow_double/`, `yellow_single/` but **all are empty**
- 4,502 rows in `manifests_rebased/crpd_yellow_train_only.csv` reference non-existent paths
- yellow_phase1 manifests also reference CRPD_raw paths (test_real.csv, val_real.csv, etc.)
- **Action**: This source MUST NOT be loaded into any pipeline

## 3. Detailed Source Breakdown

### 3.1 CCPD2019

Total in manifest (all_posquad.csv, tilt+db+challenge): **90,258** rows
Raw subdirectory counts (all subdirs):
  - ccpd_base: 199,996
  - ccpd_challenge: 50,003
  - ccpd_tilt: 30,216
  - ccpd_fn: 20,967
  - ccpd_blur: 20,611
  - ccpd_db: 10,132
  - ccpd_rotate: 10,053
  - ccpd_weather: 9,999
  - ccpd_np: 3,036

Label source: CCPD filename encoding (province+chars embedded in filename)
Existing manifest extracts text and has YOLOv8n-pose quad annotations
Family: ~100% normal7 (blue 7-char, province + alpha + 5 chars)
Quad source: `pose_yolov8n` — usable for perspective warp to 224×72
**Recommended use**: `blue` class training (real full-frame data)

### 3.2 CCPD2020 (ccpd_green)

Total: 11,776 images
  - train: 5,769
  - val: 1,001
  - test: 5,006

Label source: CCPD filename encoding → existing manifest extracts text
Family: ~100% green8 (8-char)
Sub-types: green_small (2nd char D/F), green_large (last char D/F)
Quad source: `ccpd_filename` or `pose_yolov8n` — usable for perspective warp
**Recommendation**: Train green_small/green_large. Small_old (新能源小型车) and Large (新能源大型车) distinction matters

### 3.3 CRPD_all

Breakdown by subdir:
  - CRPD_double/test: 1,102
  - CRPD_double/train: 4,000
  - CRPD_double/val: 1,000
  - CRPD_multi/test: 335
  - CRPD_multi/train: 1,000
  - CRPD_multi/val: 250
  - CRPD_single/test: 1,070
  - CRPD_single/train: 20,000
  - CRPD_single/val: 5,000

| Layout | Description | Blue (7-char) | Police | School/Trailer | Green |
|--------|------------|:-------------:|:-----:|:--------------:|:----:|
| CRPD_single | 1 plate per image | ~19,578 | ~361 | ~29 | ~2 |
| CRPD_double | 2 plates per image | ~7,938 | ~33 | ~19 | ~3 |
| CRPD_multi | 3+ plates per image | ~3,192 | ~10 | ~6 | ~1 |

Label source: YOLO label format in `labels/*.txt` — `x1 y1 x2 y2 x3 y3 x4 y4 class_id plate_text`
Format: Full-frame 1920×1080 (variable), has quad annotations in YOLO labels
Province bias: ~>99% 粤 (Guangdong) — strong regional bias
**Recommendation**: Blue class training (real road images). Do NOT use for province generalization.

### 3.4 git_plate

Total: 63,196 files (45,422 unique plate texts)
Format: Cropped plates, small resolution (typical 50×30 to 140×40)
No quad annotations — already cropped
Has augmentation variants (_distort, _stretch)

| Class | Count | Notes |
|-------|:-----:|-------|
| blue | 56,862 | |
| school_trailer_other | 1,770 | |
| police | 1,425 | |
| hk_macau | 1,418 | |
| green_large | 1,006 | |
| unknown_or_bad | 460 | |
| embassy_shi | 234 | |
| green_small | 21 | |

**Recommendation**:
- `blue`: cross-source validation only (down-weighted, small resolution)
- `police`: **cross-source validation / hard negative only** (1,425 real crops but **pre-applied warping** — images are heavily distorted; do NOT use as primary training)
- `embassy_shi`: **cross-source validation / hard negative only** (234 real crops but **pre-applied warping** — images are heavily distorted; do NOT use as primary training)
- `consulate_ling`: supplement training (~273)
- `school_trailer_other`: supplement training (~1,770)
- `hk_macau`: supplement training (1,418)
- `green_small/green_large`: supplement training (1,027)

### 3.5 CBLPRD-330k_v1

Total: 342,110 images
Format: Cropped 128×48, GAN-generated (synthetic)
Fixed resolution — no quad annotations, already cropped

| data.txt Type | Count | Text Len | Classifier Label |
|--------------|:-----:|:--------:|-----------------|
| 普通蓝牌 | 78,960 | 7 | blue |
| 新能源小型车 | 78,945 | 8 | green_small |
| 新能源大型车 | 52,630 | 8 | green_large |
| 单层黄牌 | 52,630 | 7 | yellow_single |
| 双层黄牌 | 26,315 | 7 | yellow_double (except 挂 → school_trailer) |
| 黑色车牌 | 26,315 | 7 | black (except 使→embassy, 领→consulate) |
| 拖拉机绿牌 | 26,315 | 8 | green_large (except 学→school_trailer) |

Special chars from text:
  - 使: 0 (from 黑色车牌 → embassy_shi)
  - 领: 0 (from 黑色车牌 → consulate_ling)
  - 学: 12,566 (from 单层黄牌+拖拉机绿牌 → school_trailer_other)
  - 挂: 9,893 (from 双层黄牌 → school_trailer_other)

**Critical**: CBLPRD is GAN-generated with fixed 128×48 size. Visual quality differs from real.
**Recommendation**: Heavy down-weighting (0.2×). Use for: blue, green, yellow, black classes

### 3.6 special_split_v2_20260601

| Split | Police | Embassy_shi | Total |
|-------|:-----:|:----------:|:----:|
| train | 15,500 | 10,000 | 25,500 |
| val_clean | 1,550 | 1,000 | 2,550 |
| val_hard | 1,550 | 1,000 | 2,550 |

Format: Full-frame CCPD2019 base w/ CV-replaced plate, has quad annotations
Quality: Synthetic but realistic (L-only brightness transfer + degradation)
Police: 各省均衡 (31 provinces, 500 each train)
Embassy: 10,000 train, all starting with '使'
**No consulate_ling (领) in this source** — embassy OCR keys only support 使
**Recommendation**: Primary training for police and embassy_shi

## 4. Proposed Label Taxonomy (12 Classes)

| # | Class | Description | Primary Training Sources | Cross-Val Sources |
|---|-------|-------------|:-----------------------:|:-----------------:|
| 0 | blue | 标准蓝牌 7-char | CCPD2019, CRPD_all | git_plate, CBLPRD |
| 1 | green_small | 新能源小型车 8-char D2 | CCPD2020 | git_plate, CBLPRD |
| 2 | green_large | 新能源大型车 8-char D8 | CCPD2020 | git_plate, CBLPRD |
| 3 | yellow_single | 单层黄牌 | CBLPRD | git_plate |
| 4 | yellow_double | 双层黄牌 | CBLPRD | git_plate |
| 5 | police | 警牌 (末尾警) | special_v2 | git_plate, CRPD(少量) |
| 6 | embassy_shi | 使馆牌 (使开头) | special_v2 | git_plate, CBLPRD(黑牌) |
| 7 | consulate_ling | 领馆牌 (领开头) | CBLPRD(黑牌) | git_plate |
| 8 | black | 黑色车牌 | CBLPRD(黑牌) | git_plate(少量) |
| 9 | hk_macau | 港澳牌 (含港/澳) | git_plate, CBLPRD | — |
| 10 | school_trailer_other | 学/挂/民航等 | git_plate, CBLPRD | CRPD(少量) |
| 11 | unknown_or_bad | 未知/坏样本 | (exclude) | — |

## 5. Label Mapping Rules

### 5.1 Text-based Classification
Applies to all sources that have plate text (all 6 sources):
```
1. text ends with '警' → police
2. text starts with '使' → embassy_shi
3. text starts with '领' → consulate_ling
4. text ends with '学' or '挂' → school_trailer_other
5. text contains '港' or '澳' → hk_macau
6. len=8, 2nd char D/F → green_small
7. len=8, last char D/F → green_large
8. len=8, otherwise → green_large (fallback)
9. len=7 → blue (default; override by type below)
10. other → unknown_or_bad
```

### 5.2 Source-specific Overrides

**CBLPRD**: data.txt `type` field takes precedence over text classification:
```
普通蓝牌 + len=7 → blue
新能源小型车 + len=8 → green_small
新能源大型车 + len=8 → green_large
单层黄牌 + len=7 → yellow_single (unless ends with 学→school_trailer)
双层黄牌 + len=7 → yellow_double (unless ends with 挂→school_trailer)
黑色车牌 + len=7 → black (unless starts with 使/领→embassy/consulate)
拖拉机绿牌 + len=8 → green_large (unless ends with 学→school_trailer)
```

**git_plate**:
```
Text with 'HK'/'MO' in body → blue (NOT hk_macau)
Only explicit 港/澳 character or 粤Z...港/澳 pattern → hk_macau
民航闽... → school_trailer_other
```

**CCPD2019/CCPD2020**:
```
Use existing manifest text field directly
CCPD2019 → blue, CCPD2020 → green (size distinction from text)
```

## 6. Split Design

### Proposed Training/Validation Splits

| Split | Purpose | Size Target | Class Balance | Sources Used |
|-------|---------|:-----------:|:-------------:|-------------|
| train | Primary training | ~150K | Balanced per class | All sources, capped/sampled |
| val_clean | Clean validation | 2K-3K | Balanced | All sources, base-id held out |
| val_hard | Degradation test | 1K-2K | Balanced | All sources + aug |
| val_cross_source | Domain generalization | ~1K/source | Full | Each held-out source subset |
| final_holdout | ~50 real photos | ~50 | police+embassy | User-provided |

### Constraints

1. **No base image overlap** between train and any val split — use `base_id` dedup
2. CBLPRD (GAN) → train + val_cross_source ONLY (never val_clean/hard)
3. User's real photos → final_holdout ONLY (never train)
4. val_cross_source: each source must contribute held-out samples
5. Train/val_clean/val_hard must be mutually exclusive by base_id

## 7. Source Weighting

| Source | Relative Weight | Rationale |
|--------|:--------------:|----------|
| CCPD2019 (blue) | 1.0× | Real full-frame, diverse geometry |
| CCPD2020 (green) | 3.0× | Scarce real green data, high value |
| CRPD_all (blue) | 1.0× | Real full-frame, but 粤 bias |
| special_v2 (police/emb) | 2.0× | Synthetic but well-crafted |
| git_plate (mixed) | 0.5× | Real cropped, very low resolution |
| CBLPRD (GAN) | 0.2× | GAN artifacts, uniform scale |

Without weighting, CBLPRD (342K) = 53% of total and would dominate training.

## 8. Dirty Data & Known Risks

1. **CRPD_all 粤 province bias**: ~99% from Guangdong. Use for texture variety, not region generalization.
2. **CBLPRD GAN artifacts**: 128×48 fixed resolution, uniform lighting. May need extra augmentation to match real board crop distribution.
3. **git_plate low resolution**: Many < 60×30 → resize to 224×72 with significant interpolation artifacts.
4. **consulate_ling (领) has NO synthetic dedicated dataset**: Must use CBLPRD black-plate + git_plate only.
5. **civil_aviation (民航)**: ~460 samples in git_plate, mapped to school_trailer_other.
6. **HK/Macau ambiguity**: Text body with 'HK' (e.g., 京HK3701) is standard blue, NOT hk_macau. Only 粤Z...港/澳 + explicit 港/澳 char → hk_macau.
7. **CRPD_raw_ccpd_board_v1 manifests still exist**: 4,502 broken rows. Pipeline must actively filter them.

## 9. Manifest Field Design

```csv
img_path,label,label_name,plate_text,source,source_split,crop_mode,original_w,original_h,is_synthetic,is_gan,base_id,notes
```

| Field | Description | Examples |
|-------|-------------|---------|
| img_path | Relative path from LPRNET_ROOT | datasets/CCPD2019/ccpd_tilt/0042.jpg |
| label | Integer class index (0-10) | 0 (blue), 5 (police) |
| label_name | Class string | blue, police, embassy_shi |
| plate_text | Full OCR ground truth text | 皖AEM737 |
| source | Dataset identifier | ccpd2019, ccpd2020, crpd_single, git_plate, cblprd, special_v2 |
| source_split | Original split from source | train, val, test |
| crop_mode | How to produce 224×72 | perspective_warp, resize_pad |
| original_w | Original image width | 1920, 128, etc. |
| original_h | Original image height | 1080, 48, etc. |
| is_synthetic | True if CV-replace or GAN | True for special_v2, CBLPRD |
| is_gan | True if GAN-generated | True for CBLPRD |
| base_id | Unique base image ID for dedup | CCPD2019 image stem, CBLPRD MD5 |
| notes | Free text annotation | has_aug_variant, low_res, 粤_bias |

## 10. QA Preview

Contact sheets generated at: `/home/wzzz/LPRNet/datasets/plate_classifier_audit_20260602/qa`

- `blue__cblprd.png`
- `blue__ccpd2019.png`
- `blue__crpd_CRPD_double.png`
- `blue__crpd_CRPD_multi.png`
- `blue__crpd_CRPD_single.png`
- `blue__git_plate.png`
- `class_blue.png`
- `class_embassy_shi.png`
- `class_green_large.png`
- `class_green_small.png`
- `class_hk_macau.png`
- `class_police.png`
- `class_school_trailer_other.png`
- `class_unknown_or_bad.png`
- `embassy_shi__git_plate.png`
- `embassy_shi__special_v2.png`
- `green_large__cblprd.png`
- `green_large__ccpd2020_green.png`
- `green_large__crpd_CRPD_double.png`
- `green_large__crpd_CRPD_multi.png`
- `green_large__git_plate.png`
- `green_small__cblprd.png`
- `green_small__ccpd2020_green.png`
- `green_small__git_plate.png`
- `hk_macau__cblprd.png`
- `hk_macau__git_plate.png`
- `police__crpd_CRPD_multi.png`
- `police__crpd_CRPD_single.png`
- `police__git_plate.png`
- `police__special_v2.png`
- `school_trailer_other__cblprd.png`
- `school_trailer_other__crpd_CRPD_double.png`
- `school_trailer_other__crpd_CRPD_multi.png`
- `school_trailer_other__crpd_CRPD_single.png`
- `school_trailer_other__git_plate.png`
- `source__cblprd.png`
- `source__ccpd2019.png`
- `source__ccpd2020_green.png`
- `source__crpd_CRPD_double.png`
- `source__crpd_CRPD_multi.png`
- `source__crpd_CRPD_single.png`
- `source__git_plate.png`
- `source__special_v2.png`
- `unknown_or_bad__git_plate.png`

## 11. Next Steps

1. Review QA contact sheets (each class × source)
2. Confirm label taxonomy and mapping rules
3. Build manifest generator script
4. Smoke test with minimal balanced subset (overfit64 → val_clean→100%)
5. Train classifier (blocked until approval)
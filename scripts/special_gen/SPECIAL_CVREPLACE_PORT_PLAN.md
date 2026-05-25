# Special CCPD2019 Base CV-Replace Port Plan

## What This Repo Can Reuse

This repository is a plate renderer, not a real-image replacement pipeline.
The useful reusable parts are:

- `generate_multi_plate.py`
  - `cv_imread(...)`: Chinese-path-safe image loading.
  - `get_location_data(...)`: canonical character box layout on a standard plate canvas.
  - `copy_to_image_multi(...)`: rasterizes glyphs onto a plate background with red/black/white text rules.
  - `MultiPlateGenerator.generate_plate_special(...)`: renders one requested plate string onto one requested board style.
- `plate_number.py`
  - Province list.
  - Digit and letter sets.
  - Existing special suffix glyph support such as `警` and `使`.
- Assets
  - `plate_model/yellow_140.PNG`
  - `plate_model/white_140.PNG`
  - `plate_model/black_shi_140.PNG`
  - `font_model/140_警.jpg`
  - `font_model/140_使.jpg`

These are enough to render clean canonical plate patches for:

- single-layer yellow plate
- police plate
- embassy plate, if the target style matches the existing `black_shi` board

## What This Repo Does Not Have

This repository does not contain:

- CCPD source-image loading
- pose quad or GT quad ingestion
- perspective warp from source image
- source-style transfer from real plate patch
- paste-back blending
- manifest writing
- train/val source split logic
- unique source scheduling
- QA contact sheet generation
- generation metadata summaries

Do not try to bolt all of that into `generate_special_plate.py`.
The correct place for the end-to-end data generator is a new LPRNet-side script such as:

- `scripts/special_ccpd2019_base_step1_generate.py`

## Current Mismatches To Fix Before Porting

### 1. Police number generation is too broad

`plate_number.py` currently uses `generate_plate_number_white()` for both police and army-like patterns.
For this project, police must always contain terminal `警`.
Do not reuse the current function unchanged.

### 2. Embassy format is likely incompatible with your target

`plate_number.py` currently returns embassy-like text as `使 + plate[1:]`.
Your plan says embassy text should be `number + 使` or follow the existing special-keys format.
This must be aligned with the downstream OCR keys before generation.

### 3. Board style for embassy may be wrong

This repo ships `plate_model/black_shi_140.PNG`.
Your plan text groups police and embassy under white-base color protection.
That is a policy mismatch. Before implementation, choose one of:

- keep the existing black embassy board and train with that
- add a new embassy board asset that matches your intended appearance

### 4. `generate_plate()` is not the function to port

`generate_multi_plate.py` contains local edits and an undefined `generate_plate_number_blue_copy`.
Treat `generate_plate_special(...)` as the stable export point, not the batch generator path.

## Recommended Port Architecture In LPRNet

Keep the renderer and the CV-replace pipeline separate.

### Module split

1. `scripts/special_ccpd2019_base_step1_generate.py`
   - dataset iteration
   - source split
   - source scheduling
   - plate text sampling
   - quad loading
   - source-style extraction
   - paste-back
   - manifest writing
   - QA reporting

2. `scripts/lib_special_plate_renderer.py`
   - minimal extracted code from this repo
   - canonical rendering API only

3. optional `scripts/lib_cvreplace_utils.py`
   - warp helpers
   - LAB/HSV lighting transfer
   - blur/noise estimation
   - soft-mask blending
   - color guards

### Minimal renderer API

Use a small API like:

```python
render_plate(
    text: str,
    family: str,   # yellow_single | police | embassy
) -> dict
```

Return:

- `image_bgr`: canonical plate patch, typically `440x140`
- `text`
- `family`
- `char_boxes`
- `board_style`

Do not return any source quad here. The renderer should know nothing about CCPD.

## End-to-End Generation Flow

1. Read `pose_quads.jsonl` or equivalent CCPD2019 base quad annotations.
2. Build a deterministic train/val split on source-image identity.
3. Build a unique-source scheduler:
   - shuffle each split once
   - consume each source image at most once before cycling
4. Sample target family and target text with balancing constraints.
5. Load the source image and source pose quad.
6. Warp the source plate region to a canonical plate canvas.
7. Render the target canonical plate patch from the special-plate renderer.
8. Transfer only appearance factors that should move:
   - LAB `L` statistics
   - optional low-frequency HSV `V`
   - blur level
   - noise level
9. Apply color guards:
   - yellow stays yellow
   - police stays white with visible `警`
   - embassy keeps its intended board color and visible `使`
10. Perspective-warp the synthesized patch back to the original source quad.
11. Blend with a soft edge mask.
12. Save the full image.
13. Write the manifest row using the pose quad used for OCR cropping.
14. Accumulate QA samples, coverage stats, skip stats, and family counts.

## Quad Strategy

For this project, the manifest quad should come from the pose annotation used for training-time crop.

Recommended rule:

- if `pose_quads.jsonl` exists, use it as `quad_source=pose_quad_file`
- otherwise use the best available CCPD quad and mark the source explicitly

Because the replacement is pasted back onto the same plate location, the output-image OCR quad is normally the same source quad.
That means you do not need to derive a new quad from the synthetic patch.

## Manifest Contract

Write CSV rows with at least:

- `img_path`
- `text`
- `family`
- `source`
- `split`
- `preprocess_group`
- `has_quad`
- `can_parse_ccpd_geom`
- `can_perspective`
- `quad_source`
- `bbox_source`
- `quad_1x`
- `quad_1y`
- `quad_2x`
- `quad_2y`
- `quad_3x`
- `quad_3y`
- `quad_4x`
- `quad_4y`
- `ocr_crop_mode`
- `ocr_resize_mode`
- `ocr_resize_kernel`
- `ocr_preproc`
- `ocr_channel_order`
- `ocr_quad_pad_ratio`

Recommended fixed values:

- `preprocess_group=ccpd_board`
- `has_quad=1`
- `can_parse_ccpd_geom=0`
- `can_perspective=1`
- `ocr_crop_mode=obb_warp`
- `ocr_resize_mode=letterbox`
- `ocr_resize_kernel=nn`
- `ocr_preproc=none`
- `ocr_channel_order=bgr`
- `ocr_quad_pad_ratio=0.0`

## Family-Specific Text Rules

### Yellow single

- format: `province + letter + 5 alnum`
- exclude `挂`
- exclude `学`
- single layer only

### Police

- must contain terminal `警`
- prioritize a fixed grammar rather than reusing the old white/army branch

### Embassy

- align the exact grammar with your downstream special keys first
- if using `number + 使`, add a dedicated sampler for it
- if using project-existing special format, follow that instead

## Color-Transfer Policy

Allowed transfer:

- luminance
- contrast
- low-frequency shading
- blur
- sensor noise

Disallowed transfer:

- full chroma transfer from the source plate

Practical implementation:

- match only LAB `L`
- optionally apply low-pass illumination transfer in HSV `V`
- estimate blur from the source canonical patch and match it
- estimate residual noise and add noise only after color-safe rendering

## QA Outputs Required

Per run, write:

- `generation_meta.json`
- source coverage summary
- per-family counts
- per-province counts
- skipped counts by reason
- color-guard counts
- QA contact sheet with at least 20 samples per family when available

## Implementation Order

1. Extract a minimal renderer from this repo.
2. Implement a smoke-only LPRNet generator with:
   - yellow single
   - police
   - embassy
   - pose-quad manifest writing
3. Verify train/val source-image disjointness.
4. Add luminance-only style transfer.
5. Add blur/noise transfer.
6. Add color guards and skip accounting.
7. Add contact sheet and run summary JSON.

## Bottom Line

Use this repo as an asset and canonical-rendering donor only.
Do the actual special-data generator in LPRNet, because that is where the CCPD data layout, manifest conventions, and replacement pipeline belong.

# Pseudo Anchor Guard Summary

## Goal

Use a CCPD subset that is fully disjoint from the current `train/val/test` split to create board-aligned pseudo anchors, then fine-tune LPRNet to reduce first-character confusion on:

- `粤`
- `晋`
- `黑`
- `苏`
- `浙`

The deployment baseline remains the raw board OCR path:

- `bgr`
- `crop=match`
- `resize=letterbox`
- `kernel=nn`
- `preproc=none`
- `min_occ=0.90`

## Candidate Pool Reality

The fully disjoint CCPD remainder is much smaller and more imbalanced than expected.

- Total usable extra CCPD samples outside current `train/val/test`: about `9999`
- Highly skewed toward `皖`
- Truly available focused provinces are limited:
  - `苏`: `212`
  - `浙`: `83`
  - `粤`: `26`
  - `晋`: `5`
  - `黑`: `3`

This means pseudo anchors can help `苏/浙/粤`, but they are too weak to reliably supervise `晋/黑` at scale.

## Experiment Results

| Experiment | Pseudo setup | Raw board anchor | Val exact | Test exact | Test first-char | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `first_char_guard_v1` | no CCPD pseudo anchors | correct | `92.97%` | `17.03%` | `75.57%` | current production baseline |
| `pseudo_anchor_guard_v1` | aggressive pseudo weight `192`, larger pseudo set | correct | `78.99%` | `6.48%` | `18.90%` | rejected |
| `pseudo_anchor_guard_v2` | milder pseudo weight `16`, smaller pseudo set | correct | `87.51%` | `9.38%` | `34.10%` | rejected |

## What Worked

1. The new pseudo-anchor preparation pipeline worked as intended:
   - zero overlap with existing `train/val/test`
   - board-aligned crop path matches training/inference contract
   - focused province sampling is reproducible
2. The new checkpoint-selection rule also worked:
   - real raw board anchor stayed correct
   - pseudo-anchor validation metrics affected checkpoint choice
   - later epochs were prevented from winning purely on proxy validation

## What Failed

1. Even mild pseudo-anchor pressure still pulled the global distribution away from the production-friendly checkpoint.
2. Improvements on tiny pseudo-anchor validation sets did not transfer to the large strict test split.
3. `晋` and `黑` are too scarce in the disjoint CCPD remainder, so the pseudo-anchor supervision for those classes is too fragile.

## Current Decision

Do not replace the current production model.

Keep:

- `experiments/first_char_guard_v1/weights/Final_LPRNet_model.pth`
- `experiments/first_char_guard_v1/weights/LPRNet_stage3_rk3568_fp16.rknn`

Treat `pseudo_anchor_guard_v1` and `pseudo_anchor_guard_v2` as research runs only.

## Next Recommended Step

The next useful step is not a stronger pseudo-anchor experiment.  
It is to expand the real board-side anchor pool, especially for:

- `粤`
- `晋`
- `黑`
- `苏`
- `浙`

Once real raw board anchors exist for these provinces, pseudo CCPD anchors can return as a secondary regularizer rather than a primary steering signal.

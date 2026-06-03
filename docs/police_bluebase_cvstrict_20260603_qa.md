# Police BlueBase CVStrict — QA Report

## Data: Smoke (2170 images, 31 provinces × 50/10/10)

### Generation Method
- Source: `datasets/ccpd2019_base_posquads_20260509/pose_quads.jsonl` (200K entries)
- CV features extracted per image: brightness, blur, noise, JPEG artifacts, exposure, saturation, contrast, sharpness
- Text rendered with WQY font, brightness/blur/noise/JPEG matched to source features
- Warped back to original quad, then extracted as 94x24 OCR input

### Output Files
| File | Path |
|------|------|
| Train manifest | `manifests_rebased/police_bluebase_cvstrict_20260603/train.csv` |
| Val clean manifest | `manifests_rebased/police_bluebase_cvstrict_20260603/val_clean.csv` |
| Val hard manifest | `manifests_rebased/police_bluebase_cvstrict_20260603/val_hard.csv` |
| Generation summary | `manifests_rebased/police_bluebase_cvstrict_20260603/generation_summary.json` |
| Feature stats | `manifests_rebased/police_bluebase_cvstrict_20260603/feature_stats.json` |
| Images | `datasets/police_bluebase_cvstrict_20260603/` |

### QA Checks
- [ ] 31 provinces balanced (train=50 each, val_clean=10 each, val_hard=10 each)
- [ ] Text format: province + letter + 4 alnum + 警 (7 chars)
- [ ] No I/O characters
- [ ] Images are 94x24 BGR
- [ ] No obvious misalignment
- [ ] No over/under exposure
- [ ] Text readable
- [ ] No jing at position 1 in any image

### Verdict
- [ ] PASS — proceed to formal training
- [ ] FAIL — fix generation script before training

## Formal Training Commands

See `experiments/police_bluebase_cvstrict_20260603/run_strictcv_training.sh`

Stage A (frozen backbone): 20 epoch, lr=0.001
Stage B (unfreeze .16-21): 20 epoch, lr=0.0005
Stage C (full finetune): 30 epoch, lr=0.0001

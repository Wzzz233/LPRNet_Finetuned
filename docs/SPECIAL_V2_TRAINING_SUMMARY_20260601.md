# Special V2 Training Summary — 2026-06-01

All four candidates successfully trained and evaluated. Full fine-tune with v2 data solved the bottlenecks that plagued the old frozen-backbone approach.

## Results Matrix

| Model | Warm Start | val_clean | val_hard | v1_val | Province(p0) | Tail(警) | Len Err |
|-------|:----------:|:---------:|:--------:|:------:|:------------:|:--------:|:-------:|
| **Embassy A** | special_v2 | **99.80%** | **99.70%** | **99.33%** | 100% | 99.8% | 2 |
| Embassy B | official | 99.80% | 99.10% | 99.67% | 100% | 99.8% | 1 |
| Police A | special_v2 | 98.45% | 97.10% | 98.06% | 98.77% | 99.42% | 9 |
| **Police B** | **official** | **99.48%** | **98.26%** | **99.68%** | **99.81%** | **99.61%** | 6 |

## Recommended Candidates

| Task | Candidate | Checkpoint |
|------|-----------|-----------|
| **Embassy OCR** | **A** (special warm) | `experiments/embassy_v2_fullft_specialwarm_20260601/best_LPRNet_model.pth` |
| **Police OCR** | **B** (official warm) | `experiments/police_v2_fullft_officialwarm_20260601/best_LPRNet_model.pth` |

## Comparison vs Old Models

| Metric | Old Embassy | New Embassy | Old Police | New Police |
|--------|:-----------:|:-----------:|:----------:|:----------:|
| Accuracy | 90.33% | **99.80%** | 60.00% | **99.48%** |
| Province | — | 100% | 64.52% | **99.81%** |
| Tail | — | 99.8% | 91.94% | **99.61%** |
| Len Errors | 28/300 | **2/1000** | 35/310 | **6/1550** |

## Root Cause Confirmed

The old embassy/police models failed because of **frozen backbone + limited data**. With full backbone fine-tune and v2 expanded data:
- Embassy: 3K→10K + full ft → 90%→99.8%
- Police: 3.7K→15.5K + full ft + province balance → 60%→99.5%

## Next Steps

1. ✅ Select candidates (Embassy A, Police B)
2. ⬜ Export ONNX (use FixedNorm export code)
3. ⬜ Convert to RKNN (mean=127.5, std=128, rk3568 fp16)
4. ⬜ Board deployment & verification
5. ⬜ Add val_jitter / val_boardhard dataset for real-world verification

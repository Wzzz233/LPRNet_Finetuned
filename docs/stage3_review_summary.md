# Stage 3 Review Summary

> Generated: 2026-05-07

---

## 1. Current Phase Completed

| Deliverable | Files | Status |
|------------|-------|--------|
| Updated dataset_root patch doc | `docs/dataset_root_patch.md` | ✅ |
| Migration plan risk review | `docs/migration_plan_review.md`, `migration_plan_review.json` | ✅ |
| Experiment catalog | `docs/experiment_catalog.md`, `experiment_catalog.json` | ✅ |
| Config migration suggestions | `docs/rebased_config_migration_suggestions.md`, `rebased_config_migration_suggestions.json` | ✅ |
| Stage 3 summary | `docs/stage3_review_summary.md` | ✅ |

---

## 2. Manifest Rebase Status

| Metric | Value |
|--------|-------|
| Old manifest files | 440 (376 high-risk, absolute paths) |
| Rebased manifest files | 336 (in manifests_rebased/) |
| Remaining manual review | 40 (cross-root / broken paths) |
| Absolute paths removed | 754,479 |
| Broken paths (before rebase) | 53,585 |
| Broken paths (after rebase) | 3,811 (all from green_edgefit softlink cases) |

---

## 3. Dataset Root Patch Status

| Feature | Status |
|---------|--------|
| `--dataset_root` | ✅ Added (default=".") |
| `--strict_path_check` | ✅ Added (optional) |
| `--max_steps` | ✅ Added (default 0=no limit) |
| `_resolve_img_path()` | ✅ Added (abs/rel path detection) |
| Path filter fix (train/test) | ✅ `os.path.exists()` now uses resolved path |
| Legacy backward compat | ✅ Absolute paths, old CWD, old abs manifest all work |

---

## 4. Smoke Test Status

| Test | Manifest | Root | Samples | Steps | Loss | Status |
|------|---------|------|---------|-------|------|--------|
| yellow | `manifests_rebased/yellow_train.csv` | `/home/wzzz/LPRNet` | 54,566 | 100 | 6.51 | ✅ |
| firstchar | `manifests_rebased/firstchar_tiny_gray_alldata_v1/train.csv` | `/home/wzzz/LPRNet` | 208,049 | 100 | 5.77 | ✅ |
| green_edgefit | `manifests_rebased/unified_manifest_green_edgefit_v3_allprov.csv` | `/home/wzzz/LPRNet/datasets` | 7,220 | 100 | 5.64 | ✅ |

All three confirm: dataloader, forward, loss, backward, max_stops all work.

---

## 5. Migration Plan Risk Review

| Category | Count | Action |
|----------|-------|--------|
| **safe_to_move_now** | **36** | Temp scripts, reports, `__pycache__` |
| **defer_until_experiment_catalog** | **130** | Experiments (97), checkpoints (33) |
| **blocked_high_risk** | **387** | Manifests (377), datasets (8), configs (2) |

---

## 6. First Low-Risk Items (36 items)

These are truly movable without breaking the training pipeline:

- **11 root temp scripts** → move to `tools/` (e.g. `_check_qa04.py`, `tmp_export_refiner_*.py`)
- **1 Windows Zone.Identifier** → move to `tmp/cleanup_candidates/`
- **20 `__pycache__` directories** → move to `tmp/cleanup_candidates/`
- **4 qa_* report directories** → move to `reports/qa/`

---

## 7. Deferred Items (130 items)

These require experiment catalog review first:

- **97 experiment directories** (archive candidates, no weights/logs/configs)
- **33 weight-only directories** (named `weights`, `weights_stageA/B/C`)

---

## 8. Blocked Items (387 items)

These MUST NOT be moved under any circumstances:

- **377 manifest items** (both old and rebased)
- **8 dataset symlinks** (CCPD2019, CCPD2020, CRPD_all, etc.)
- **2 config items**
- All items with `requires_manual_review=True`

---

## 9. Experiment Catalog Conclusions

| Metric | Value |
|--------|-------|
| Total experiments | 271 |
| Active (keep) | 177 |
| Archived candidates | 87 |
| Failed candidates | 7 |
| Full reproducibility chain | 40 |
| Partial reproducibility | 126 |
| Weak reproducibility | 17 |

**Key finding**: 40 experiments have full reproducibility (config+log+checkpoint+manifest+training_root).
44 experiments have matching rebased manifests ready for migration.

---

## 10. Recommended Active Experiments for Rebase Migration

Top 5 candidates from 40 eligible experiments:

| # | Experiment | Manifest | Dataset Root |
|---|-----------|---------|-------------|
| 1 | yellow_single_v1_phase1 | `manifests_rebased/yellow_single_train.csv` | `/home/wzzz/LPRNet` |
| 2 | yellow_single_v1_phase2 | `manifests_rebased/yellow_single_train.csv` | `/home/wzzz/LPRNet` |
| 3 | yellow_single_v2_weighted_phase1 | `manifests_rebased/yellow_single_train_weighted.csv` | `/home/wzzz/LPRNet` |
| 4 | yellow_single_v2_weighted_phase2 | `manifests_rebased/yellow_single_train_weighted.csv` | `/home/wzzz/LPRNet` |
| 5 | curriculum_gray3_stageA | `manifests_rebased/curriculum_gray3/train_stageA.csv` | `/home/wzzz/LPRNet` |

---

## 11. Recommended Next Steps (Ordered)

1. **Read** `docs/migration_plan_review.md` — understand the risk classification
2. **Read** `docs/experiment_catalog.md` — understand experiment reproducibility
3. **Execute** only `safe_to_move_now` items (36 low-risk moves)
4. **Pick 1 active experiment** (e.g. `yellow_single_v1_phase1`) → switch to rebased manifest
5. **Run max_steps=100 smoke test** on the switched experiment
6. **If pass, run 1 epoch validation** (no max_steps limit)
7. **If pass, gradually switch more experiments**
8. **Still don't archive experiments** — wait until all active experiments are verified

---

## 12. What NOT To Do

- ❌ Execute `migration_plan.json`
- ❌ Move `datasets/`, `data/`, `manifests/`, `manifests_rebased/`, `src/`, `configs/`
- ❌ Move or merge any weight files (`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`, `*.engine`)
- ❌ Archive experiment directories
- ❌ Delete `__pycache__`, `Zone.Identifier`, cache, tmp files
- ❌ Create green_edgefit softlink
- ❌ Replace all training configs at once
- ❌ Run full training
- ❌ Overwrite old experiment outputs
- ❌ Modify old manifests
- ❌ Modify historical experiment records

---

## File Inventory

| File | Description |
|------|-------------|
| `docs/dataset_root_patch.md` | dataset_root, strict_path_check, max_steps docs |
| `docs/migration_plan_review.md` | Risk review of all 553 migration plan items |
| `migration_plan_review.json` | Structured review data |
| `docs/experiment_catalog.md` | Experiment reproducibility catalog |
| `experiment_catalog.json` | Structured catalog data |
| `docs/rebased_config_migration_suggestions.md` | Top 5 rebase migration suggestions |
| `rebased_config_migration_suggestions.json` | Structured suggestion data |
| `docs/stage3_review_summary.md` | This summary |

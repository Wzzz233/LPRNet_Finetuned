# Migration Plan Risk Review

Date: 2026-05-07
Total plan items: 553

## Classification

| Category | Count | Description |
|----------|-------|-------------|
| safe_to_move_now | 36 | Temporary scripts, reports, non-participating files |
| defer_until_experiment_catalog | 130 | Experiments, checkpoints, weights |
| blocked_high_risk | 387 | Datasets, manifests, configs, source code |

## Safe to Move Now

The following 36 items have low dependency risk and COULD be moved safely.

However, all cleanup/move actions are currently in DRY-RUN mode.

| # | Old Path | Type | Risk | Action |
|---|----------|------|------|--------|
| 1 | _check_qa04.py | temp | low | -> tools/_check_qa04.py |
| 2 | _check_qa04_detailed.py | temp | low | -> tools/_check_qa04_detailed.py |
| 3 | _check_quad_order.py | temp | low | -> tools/_check_quad_order.py |
| 4 | _filter_replace_val.py | temp | low | -> tools/_filter_replace_val.py |
| 5 | _inspect_qa04.py | temp | low | -> tools/_inspect_qa04.py |
| 6 | _inspect_qa04_visual.py | temp | low | -> tools/_inspect_qa04_visual.py |
| 7 | _inspect_warp_quality.py | temp | low | -> tools/_inspect_warp_quality.py |
| 8 | _qa04_comparison.py | temp | low | -> tools/_qa04_comparison.py |
| 9 | tmp_export_refiner_onnx.py | temp | low | -> tools/tmp_export_refiner_onnx.py |
| 10 | tmp_export_refiner_rknn_friendly.py | temp | low | -> tools/tmp_export_refiner_rknn_friendly.p |
| 11 | tmp_export_refiner_static_onnx.py | temp | low | -> tools/tmp_export_refiner_static_onnx.py |
| 12 | true_quad_refiner_robust_plan.md:Zone.Identifier | temp | low | -> (cleanup candidate) |
| 13 | __pycache__ | temp | low | -> (cleanup candidate) |
| 14 | artifacts/green_board_vs_training_demo/__pycache__ | temp | low | -> (cleanup candidate) |
| 15 | data/__pycache__ | temp | low | -> (cleanup candidate) |
| 16 | scripts/__pycache__ | temp | low | -> (cleanup candidate) |
| 17 | scripts/analysis/__pycache__ | temp | low | -> (cleanup candidate) |
| 18 | scripts/curriculum_gray3/__pycache__ | temp | low | -> (cleanup candidate) |
| 19 | tests/__pycache__ | temp | low | -> (cleanup candidate) |
| 20 | tmp_scripts/__pycache__ | temp | low | -> (cleanup candidate) |
| 21 | src/__pycache__ | temp | low | -> (cleanup candidate) |
| 22 | src/quad_refiner/__pycache__ | temp | low | -> (cleanup candidate) |
| 23 | src/training/__pycache__ | temp | low | -> (cleanup candidate) |
| 24 | src/evaluation/__pycache__ | temp | low | -> (cleanup candidate) |
| 25 | src/export/__pycache__ | temp | low | -> (cleanup candidate) |
| 26 | src/manifest/__pycache__ | temp | low | -> (cleanup candidate) |
| 27 | src/utils/__pycache__ | temp | low | -> (cleanup candidate) |
| 28 | src/micro_rectifier/__pycache__ | temp | low | -> (cleanup candidate) |
| 29 | models/weights/core/__pycache__ | temp | low | -> (cleanup candidate) |
| 30 | tmp/__pycache__ | temp | low | -> (cleanup candidate) |
| 31 | tmp/e1_v4_hard_proxy_qa_20260411/__pycache__ | temp | low | -> (cleanup candidate) |
| 32 | qa_firstchar_hires_rawcrop | report | low | -> reports/qa/qa_firstchar_hires_rawcrop |
| 33 | qa_firstchar_hires_rawcrop_abc | report | low | -> reports/qa/qa_firstchar_hires_rawcrop_ab |
| 34 | qa_firstchar_patch_ab | report | low | -> reports/qa/qa_firstchar_patch_ab |
| 35 | qa_firstchar_patch_ab_v2 | report | low | -> reports/qa/qa_firstchar_patch_ab_v2 |
| 36 | qa_u1c_green_generation_difficulty | report | low | -> reports/qa/qa_u1c_green_generation_diffi |

## Deferred Until Experiment Catalog

The following 130 items involve experiments, checkpoints, or weights.

These are deferred until the experiment catalog is reviewed.

| File Type | Count |
|-----------|-------|
| experiment | 97 |
| checkpoint | 33 |

## Blocked High Risk

The following 387 items involve protected resources.

These are BLOCKED and must not be executed.

| File Type | Count |
|-----------|-------|
| manifest | 377 |
| dataset | 8 |
| config | 2 |

## Blocked Items Detail

| # | Old Path | Type | Reason |
|---|----------|------|--------|
| 1 | manifests/Archive | manifest | manifests/Archive 标准化命名 |
| 2 | manifests/Archive/unified_manifest_green_balance_aggr_v | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 3 | manifests/Archive/unified_manifest_green_balance_aggr_v | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 4 | manifests/Archive/unified_manifest_green_balance_aggr_v | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 5 | manifests/Archive/unified_manifest_green_balance_baseli | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 6 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 7 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 8 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 9 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 10 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 11 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 12 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 13 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 14 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 15 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 16 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 17 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 18 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 19 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |
| 20 | manifests/Archive/unified_manifest_green_balance_round2 | manifest | MANIFEST 含 3331 个绝对路径，需要生成新相对路径版本替代。不要原地修改 |

## Weight Items Detail

Weight items: 33 (all deferred or blocked)

- experiments/crop_aligned_v1/weights [defer_until_catalog]

- experiments/first_char_guard_v1/weights [defer_until_catalog]

- experiments/first_board_baseline_v1/weights [defer_until_catalog]

- experiments/green_official_v4_prod_20260320_192917/weig [defer_until_catalog]

- experiments/green_official_v5a_prod_20260320_193511/wei [defer_until_catalog]

- experiments/green_official_v5b_prod_20260320_194901/wei [defer_until_catalog]

- experiments/green_official_v5c_prod_20260320_195537/wei [defer_until_catalog]

- experiments/green_specialist_official_debug_20260320_11 [defer_until_catalog]

- experiments/green_specialist_official_debug_20260320_11 [defer_until_catalog]

- experiments/green_specialist_official_debug_20260320_11 [defer_until_catalog]


## Summary Statistics

- Total migration plan items: 553
- safe_to_move_now: 36
- defer_until_experiment_catalog: 130
- blocked_high_risk: 387
- Weight items: 33
- Manifest items: 377
- Dataset items: 8
- Source/Config items: 2
- Delete/cleanup actions (pycache, Zone.Identifier): 20 (all moved to cleanup_candidates bucket)

## Execution Policy

1. **Execution NOT permitted** — all in dry-run/review mode
2. All 'delete' actions changed to 'move to tmp/cleanup_candidates/'
3. No weight merging until experiment catalog is complete
4. No experiment archiving until experiment catalog is complete
5. No manifest, config, dataset, or source file moves
6. Safe items require manual confirmation before any execute

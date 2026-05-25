# Root Tidy Execute Report

> Date: 2026-05-07
> Command: `python tools/cleanup/tidy_workspace_root.py --plan root_tidy_plan.json --execute`

## Summary

| Metric | Value |
|--------|-------|
| Plan items | 34 |
| Moved | 34 |
| Blocked (core assets) | 0 |
| Skipped (would overwrite) | 0 |
| Overwrite risk | None |
| Core assets affected | None |

## Moved to `tmp/generated_json/` (21 files)

| File | Size |
|------|------|
| experiment_catalog.json | JSON |
| expert_archive_plan.json | JSON |
| expert_asset_candidates.json | JSON |
| green_plate_expert_completion_plan.json | JSON |
| blue_plate_expert_completion_plan.json | JSON |
| manifest_path_inventory_rebased.json | JSON |
| manifest_rebase_dry_run.json | JSON |
| manifest_rebase_execute_report.json | JSON |
| manifest_rebase_plan.json | JSON |
| migration_plan.json | JSON |
| migration_plan_review.json | JSON |
| model_zoo_completeness_review.json | JSON |
| next_rebased_representative_experiments.json | JSON |
| rebased_config_migration_suggestions.json | JSON |
| rebased_dataset_root_consistency.json | JSON |
| rebased_manifest_training_roots.json | JSON |
| rebased_training_smoke_test.json | JSON |
| yellow_single_batch_smoke_result.json | JSON |
| yellow_single_rebased_batch_plan.json | JSON |
| yellow_single_v1_phase1_rebased_epoch1_result.json | JSON |
| yellow_single_v1_phase1_rebased_smoke_result.json | JSON |

## Moved to `tools/` (11 files)

| File | Type |
|------|------|
| _check_qa04.py | temporary script |
| _check_qa04_detailed.py | temporary script |
| _check_quad_order.py | temporary script |
| _filter_replace_val.py | temporary script |
| _inspect_qa04.py | temporary script |
| _inspect_qa04_visual.py | temporary script |
| _inspect_warp_quality.py | temporary script |
| _qa04_comparison.py | temporary script |
| tmp_export_refiner_onnx.py | temporary script |
| tmp_export_refiner_rknn_friendly.py | temporary script |
| tmp_export_refiner_static_onnx.py | temporary script |

## Moved to `tmp/cleanup_candidates/` (2 files)

| File | Type |
|------|------|
| true_quad_refiner_robust_plan.md:Zone.Identifier | Windows metadata |
| hermes_conversation_20260424_193850.json | Conversation cache |

## Verification

- Snapshot before: 103 root entries
- Snapshot after: 70 root entries
- All 34 moved files confirmed by diff
- Rollback plan available: `root_tidy_rollback_plan.json`

## Safety Confirmation

- ✅ No datasets moved
- ✅ No experiment directories moved
- ✅ No manifests modified or removed
- ✅ No configs changed
- ✅ No checkpoints or board artifacts touched
- ✅ No files deleted (all items moved to organized locations)
- ✅ Full rollback possible via `root_tidy_rollback_plan.json`

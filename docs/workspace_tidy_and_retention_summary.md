# Workspace Tidy and Retention Summary

## 1. Model Zoo Status

| Expert | Status | Keys | Manifest | Checkpoint | Board Artifact |
|--------|--------|------|----------|------------|----------------|
| yellow_from_blue_expert | archive_partial | ✅ yellow_keys | ✅ rebased | ref only | ref only |
| special_plate_expert | archive_partial | ✅ special_keys | ✅ rebased | ref only | ref only |
| blue_plate_expert | archive_partial | ✅ default CHARS | txt reference | ref only | ref + ONNX + RKNN |
| green_plate_expert | archive_partial | ✅ default CHARS | ✅ rebased | ref only | ref + ONNX + RKNN |

- All 4 experts have README.md, lineage.json, keys, manifests in model_zoo/
- Checkpoints and board artifacts remain as reference paths (not copied)
- Yellow expert is the most complete (rebased_verified, smoke/epoch verified)
- Blue expert needs rebased CSV manifest identification
- Green expert prov_deg model confirmed at experiments/green_e12_province_degrade_unfreeze/

## 2. Root Tidy Plan

| Category | Count |
|----------|-------|
| Movable (can_move_now) | 34 |
| Blocked (core assets) | 0 |
| Would overwrite | 0 |

Movable items:
- 21 intermediate JSON files → tmp/generated_json/
- 11 temporary Python scripts → tools/
- 2 cleanup candidates → tmp/cleanup_candidates/

## 3. No Core Assets Moved

- All datasets, manifests, rebased manifests, experiments, src, configs, model_zoo remain in place.
- Zero weight, ONNX, RKNN, board artifact files are moved.
- Zero core directories are moved.

## 4. Manifest Retention

| Category | Count |
|----------|-------|
| keep_expert_core | 13 |
| keep_active_reference | 2 |
| keep_legacy_reference | 1 |
| cleanup_candidate | 1 |
| manual_review | 3 |

## 5. Dataset Retention

| Category | Count |
|----------|-------|
| keep_expert_core_dataset | 8 |
| keep_legacy_dataset | 22 (needs further review) |
| cleanup_candidate_dataset | ~23 small QA/probe datasets |

## 6. Next Steps (Recommended Order)

1. Review `docs/root_tidy_plan.md` to confirm no core assets affected
2. Run: `python tools/cleanup/tidy_workspace_root.py --plan root_tidy_plan.json --execute`
3. Move only root-level JSON, temp scripts, and cleanup candidates
4. Do NOT move datasets, manifests, experiments, checkpoints, or board artifacts
5. Do NOT delete anything — all items are moved, not removed
6. Later: when model_zoo needs delivery, run build_expert_archives with --copy-checkpoints
7. Much later: review manifest cleanup candidates (only after expert archives don't need old paths)

## 7. What Was NOT Done

- ❌ No files deleted
- ❌ No datasets moved
- ❌ No experiment directories moved
- ❌ No manifests modified or removed
- ❌ No configs changed
- ❌ No checkpoints or board artifacts moved
- ❌ No full training runs
- ❌ No migration_plan execution
- ❌ No symlinks created

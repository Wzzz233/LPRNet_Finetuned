# Preflight Check: yellow_single_v1_phase1 Rebased Smoke Test

Script: configs/rebased_validation/yellow_single_v1_phase1_rebased_smoke.sh

## Check Results

| Check | Status | Note |
|-------|--------|------|
| Uses manifests_rebased/ in training command | ✅ | `manifests_rebased/yellow_train.csv` |
| Old manifest not referenced in command body | ✅  | 只在 Rollback Notes 中提及 |
| Has `--dataset_root /home/wzzz/LPRNet` | ✅ | 正确 |
| Has `--max_steps 100` | ✅ | 正确 |
| Output to `rebased_validation/` | ✅ | `experiments/rebased_validation/` |
| Does NOT write to old experiment dir | ✅ | 命令体不含旧路径 |
| No dangerous commands (rm/mv/cp) | ✅ | 无 |
| Has rollback notes | ✅ | 有 |
| `--pretrained_model` is empty | ✅ | 随机初始化，不加载旧权重 |
| Old experiment dir exists | ✅ | 未受影响 |
| New output dir does not exist (clean) | ✅ | 即将创建 |

## Summary

- All 11 checks passed. False positives for 2 checks were rollback notes, not training commands.
- **Can proceed with smoke test: YES**

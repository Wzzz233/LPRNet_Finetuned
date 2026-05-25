# dataset_root Patch 说明

## 为什么需要 dataset_root

原 `train_LPRNet.py` + `UnifiedManifestDataset` 使用 manifest 中的 `img_path` 直接调用 `cv2.imread()`。
路径解析完全依赖于当前工作目录 (CWD)：

```python
# 旧逻辑 (load_data.py:913)
image = cv2.imread(row['img_path'])
```

对于旧的绝对路径 manifest（如 `/home/wzzz/LPRNet/datasets/xxx.jpg`），这不构成问题。
但 rebased manifest 的路径是相对于项目根目录的：

```
旧: /home/wzzz/LPRNet/CCPD2020/ccpd_green/train/xxx.jpg
新: CCPD2020/ccpd_green/train/xxx.jpg
```

当训练从 `src/training/` 目录启动时，CWD = `/home/wzzz/LPRNet/src/training/`，
`cv2.imread('CCPD2020/xxx.jpg')` 会查找 `/home/wzzz/LPRNet/src/training/CCPD2020/xxx.jpg` → **找不到**。

## 改动范围

### 1. `src/training/train_LPRNet.py`

新增参数解析：
```python
parser.add_argument('--dataset_root', default='.',
    help='dataset root directory; relative manifest img_path entries '
         'are resolved relative to this. set to /home/wzzz/LPRNet for rebased manifests')
parser.add_argument('--strict_path_check', action='store_true',
    help='if set, raise error immediately when a manifest image path does not exist')
parser.add_argument('--max_steps', default=0, type=int,
    help='max training steps (batches); 0=no limit. stops training after '
         'this many batches across all epochs')
```

新增到 `common_dataset_kwargs`：
```python
common_dataset_kwargs = dict(
    ...,
    dataset_root=os.path.expanduser(args.dataset_root),
    strict_path_check=args.strict_path_check,
)
```

路径过滤逻辑（train 和 test 的 `os.path.exists()` 检查）已改为使用 `_resolve_img_path()`：
```python
# 旧:
if img_path and os.path.exists(img_path):
# 新:
if img_path:
    resolved = train_main_dataset._resolve_img_path(img_path)
    if os.path.exists(resolved):
```

### 2. `src/load_data.py`

`UnifiedManifestDataset.__init__` 新增参数：
```python
self.dataset_root = os.path.expanduser(dataset_root) if dataset_root else '.'
self.strict_path_check = strict_path_check
```

新方法 `_resolve_img_path()`：
```python
def _resolve_img_path(self, raw_path):
    if os.path.isabs(raw_path):
        return raw_path          # 绝对路径：不拼接
    return os.path.join(self.dataset_root, raw_path)  # 相对路径：拼接
```

`__getitem__` 的修改：
- 使用 `_resolve_img_path()` 解析路径后传参给 `cv2.imread()`
- 错误日志同时显示 resolved_img_path 和 raw_img_path
- `strict_path_check` 为 False 时自动 fallback 尝试原始路径（兼容旧 manifest）

### 3. 总改动行数

| 文件 | 净增行数 | 说明 |
|------|---------|------|
| `src/training/train_LPRNet.py` | +17 | argparse、max_steps 检查、路径过滤 |
| `src/load_data.py` | +11 | dataset_root、_resolve_img_path、__getitem__ 路径解析 |
| 总计 | +28 | |

## 参数说明

### --dataset_root (default=".")

相对路径 manifest 的根目录。设为 `/home/wzzz/LPRNet` 适配 rebased manifest。
旧配置不传此参数时默认为当前工作目录 (CWD)，保持向后兼容。

### --strict_path_check (action)

默认关闭。关闭时保持原有容错逻辑（跳过不存在路径的行）。
开启后遇到找不到的路径直接报错中断，适合调试。

### --max_steps (default=0)

最大训练步数（batch 数）。0 表示不限制。
支持外层 epoch 循环正确退出（双 break 机制）。

## UnifiedManifestDataset 路径解析规则

```
输入: row['img_path'] (来自 manifest CSV)
  │
  ├── os.path.isabs(img_path) == True
  │     → 使用原路径（兼容旧绝对路径 manifest）
  │
  └── os.path.isabs(img_path) == False
        → os.path.join(self.dataset_root, img_path)
        → 使用拼接后的路径
```

## rebased manifest 的训练路径契约

### 大多数 rebased manifest

```
dataset_root = /home/wzzz/LPRNet
```

示例：
```bash
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet \
  --train_manifest manifests_rebased/yellow_train.csv \
  --test_manifest manifests_rebased/yellow_real_val.csv \
  ...
```

### green_edgefit_v3_allprov（特殊 case）

```
dataset_root = /home/wzzz/LPRNet/datasets
```

原因：路径以 `green_edgefit_v3_allprov/` 开头，没有 `datasets/` 前缀，
而数据实际位于 `datasets/green_edgefit_v3_allprov/`。

```bash
python src/training/train_LPRNet.py \
  --dataset_root /home/wzzz/LPRNet/datasets \
  --train_manifest manifests_rebased/unified_manifest_green_edgefit_v3_allprov.csv \
  ...
```

**不建议创建 green_edgefit 根目录软链接**，因为：
- `dataset_root=/home/wzzz/LPRNet/datasets` 已经可以工作
- 创建软链接会增加旧路径兼容层的维护成本
- 如果以后迁移到新机器，`dataset_root` 配置比软链接更可靠

## Smock Test 结果

| 测试 | Manifest | Dataset Root | 样本数 | Batch | Steps | Avg Loss | 状态 |
|------|---------|-------------|-------|-------|-------|---------|------|
| yellow | `manifests_rebased/yellow_train.csv` | `/home/wzzz/LPRNet` | 54,566 | 4 | 100 | 6.51 | ✅ |
| firstchar | `manifests_rebased/firstchar_tiny_gray_alldata_v1/train.csv` | `/home/wzzz/LPRNet` | 208,049 | 4 | 100 | 5.77 | ✅ |
| green_edgefit | `manifests_rebased/unified_manifest_green_edgefit_v3_allprov.csv` | `/home/wzzz/LPRNet/datasets` | 7,220 | 4 | 100 | 5.64 | ✅ |

- ✅ Dataloader 返回 batch
- ✅ Forward 正常
- ✅ Loss 计算正常
- ✅ Backward 正常
- ✅ max_steps=100 正确停止
- ✅ 输出到 `experiments/validation_rebased_manifest/`（无覆盖风险）

## 兼容性

- **旧 manifest（绝对路径）**：不传 `--dataset_root`，或传默认值 `.`，绝对路径不走拼接。
- **旧启动方式**：`cd src/training && python train_LPRNet.py` 仍然可用。
- **旧绝对路径 manifest**：`os.path.isabs()` 判断→不拼接，行为不变。
- **后续新实验**：必须优先使用 `manifests_rebased/` + `--dataset_root`。
- **旧 manifests/**：不要删除，作为 legacy reference 保留。

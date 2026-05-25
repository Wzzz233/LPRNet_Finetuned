# Rebased Manifest Validation Commands

> 生成时间: 2026-05-07
> 说明: 这些命令用于验证 rebased manifest + training_dataset_root 能正常加载数据。
> 所有命令都在 `/home/wzzz/LPRNet` 下执行。
> 不要直接覆盖已有训练实验。

---

## 前提

当前 `train_LPRNet.py` 不支持 `--dataset_root`。以下命令通过设置 CWD 为 PROJECT_ROOT 来使相对路径正确解析。如果正式使用，建议先给训练脚本添加 `--dataset_root` 支持。

---

## 1. 读取验证 (不训练)

### 1.1 yellow_train.csv (主验证)

```bash
cd /home/wzzz/LPRNet

python -c "
import csv, os
manifest = 'manifests_rebased/yellow_train.csv'
root = '/home/wzzz/LPRNet'
valid = 0
total = 0
with open(manifest) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 100: break
        total += 1
        path = os.path.join(root, row['img_path'])
        if os.path.exists(path):
            valid += 1
print(f'yellow_train: {valid}/{total} valid')
"
```

### 1.2 firstchar_tiny_gray_alldata_v1/train.csv (子目录 manifest)

```bash
cd /home/wzzz/LPRNet

python -c "
import csv, os
manifest = 'manifests_rebased/firstchar_tiny_gray_alldata_v1/train.csv'
root = '/home/wzzz/LPRNet'
valid = 0
total = 0
with open(manifest) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 100: break
        total += 1
        path = os.path.join(root, row['img_path'])
        if os.path.exists(path):
            valid += 1
print(f'firstchar: {valid}/{total} valid')
"
```

### 1.3 green_edgefit_v3_allprov.csv (datasets root 验证)

```bash
cd /home/wzzz/LPRNet/datasets

python -c "
import csv, os
manifest = '../manifests_rebased/unified_manifest_green_edgefit_v3_allprov.csv'
root = '.'  # CWD = datasets/
valid = 0
total = 0
with open(manifest) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 100: break
        total += 1
        path = os.path.join(root, row['img_path'])
        if os.path.exists(path):
            valid += 1
print(f'green_edgefit: {valid}/{total} valid (CWD hack with root=datasets/)')
"
```

---

## 2. Dataloader Smoke Test

验证 PyTorch DataLoader 能正常返回 batch。

### 2.1 yellow_train dataloader

```bash
cd /home/wzzz/LPRNet

python -c "
import sys; sys.path.insert(0, 'src')
from load_data import UnifiedManifestDataset
from torch.utils.data import DataLoader

ds = UnifiedManifestDataset(
    manifest_path='manifests_rebased/yellow_train.csv',
    img_size=[94, 24],
    lpr_max_len=8,
    split_filter='train',
)
loader = DataLoader(ds, batch_size=4, num_workers=0)
batch = next(iter(loader))
print(f'Batch loaded: images.shape={batch[0].shape}, labels={batch[1]}')
"
```

### 2.2 firstchar dataloader

```bash
cd /home/wzzz/LPRNet

python -c "
import sys; sys.path.insert(0, 'src')
from load_data import UnifiedManifestDataset
from torch.utils.data import DataLoader

ds = UnifiedManifestDataset(
    manifest_path='manifests_rebased/firstchar_tiny_gray_alldata_v1/train.csv',
    img_size=[94, 24],
    lpr_max_len=8,
    split_filter='train',
)
loader = DataLoader(ds, batch_size=4, num_workers=0)
batch = next(iter(loader))
print(f'Batch loaded: images.shape={batch[0].shape}, labels={batch[1]}')
"
```

### 2.3 green_edgefit dataloader (特殊 root)

```bash
cd /home/wzzz/LPRNet/datasets

python -c "
import sys; sys.path.insert(0, '../src')
from load_data import UnifiedManifestDataset
from torch.utils.data import DataLoader

ds = UnifiedManifestDataset(
    manifest_path='../manifests_rebased/unified_manifest_green_edgefit_v3_allprov.csv',
    img_size=[94, 24],
    lpr_max_len=8,
    split_filter='train',
)
loader = DataLoader(ds, batch_size=4, num_workers=0)
batch = next(iter(loader))
print(f'Batch loaded: images.shape={batch[0].shape}, labels={batch[1]}')
"
```

---

## 3. 最小训练 Smoke Test (最多 1 epoch, 100 step)

使用 `configs/rebased_validation/` 中的脚本。

```bash
# 测试 1: yellow (PROJECT_ROOT)
cd /home/wzzz/LPRNet
bash configs/rebased_validation/yellow_train_validation.sh

# 测试 2: firstchar (PROJECT_ROOT)  
bash configs/rebased_validation/firstchar_tiny_gray_validation.sh

# 测试 3: green_edgefit (datasets root)
bash configs/rebased_validation/green_edgefit_validation.sh
```

### 安全确认

- 输出目录: `experiments/validation_rebased_manifest/` — 新目录，不覆盖任何旧实验
- batch_size=4, epoch=1, num_workers=0 — 最小资源消耗
- 测试后检查 `smoke.log` 确认无错误
- 不设置 `--pretrained_model` — 不从已有权重初始化，不干扰已有实验

---

## 4. 确认不会覆盖已有权重的检查

```bash
ls /home/wzzz/LPRNet/experiments/validation_rebased_manifest/ 2>/dev/null
# 应该为空（第一次运行前）或只包含之前 3 个测试的输出
```

---

## 5. 验证成功后下一步

1. 给 `train_LPRNet.py` 添加 `--dataset_root` 支持
2. 更新所有实验启动脚本（`cd /home/wzzz/LPRNet && python src/training/train_LPRNet.py ...`）
3. 或保持 CWD hack（不推荐长期使用）
4. 渐进式切换实验配置引用新 manifest

---

*不要直接完整训练。先运行 dataloader smoke test 确认路径正确。*

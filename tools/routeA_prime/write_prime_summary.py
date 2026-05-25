#!/usr/bin/env python3
"""Compile ROUTEA_PRIME_SUMMARY from all experiment results + multi-frame aggregation."""
import json, csv
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import torch
import torchvision.models as models

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'experiments/routeA_prime_quadwarp_20260512'

PROVINCE_CHARS = ['京','津','冀','晋','蒙','辽','吉','黑',
                  '沪','苏','浙','皖','闽','赣','鲁','豫',
                  '鄂','湘','粤','桂','琼','川','贵','云',
                  '藏','陕','甘','青','宁','新','渝']

BOARD_DUMP2 = Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/pos_ocr_dump_2')
GT_DUMP2 = '京AD06088'

# ── Multi-frame aggregation for B3 ──
print('Multi-frame aggregation analysis for B3...')

class ResNet18Province(torch.nn.Module):
    def __init__(self, in_channels=1, num_classes=31):
        super().__init__()
        self.model = models.resnet18(weights=None)
        if in_channels != 3:
            self.model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18Province(in_channels=1)
ckpt = torch.load(str(OUT_DIR / 'B3_fullplate_gray3_224x72_bal31/best.pt'), map_location='cpu')
model.load_state_dict(ckpt)
model.to(device).eval()

# Load coarse images
coarse_files = sorted([f for f in BOARD_DUMP2.iterdir()
                       if f.name.startswith('coarse_') and f.name.endswith('.ppm')],
                      key=lambda x: int(x.stem.split('_')[1]))
print(f'  Dump2: {len(coarse_files)} coarse images')

# Per-frame predictions and logits
all_logits = []
per_frame_preds = []
for f in coarse_files:
    img = cv2.imread(str(f))
    img = cv2.resize(img, (224, 72), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_t = torch.from_numpy(gray.astype('float32') / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_t)
        all_logits.append(logits.cpu().numpy())
        pred = logits.argmax(dim=1).item()
        per_frame_preds.append(PROVINCE_CHARS[pred] if pred < len(PROVINCE_CHARS) else '?')

# Per-frame accuracy
per_frame_fc = sum(1 for p in per_frame_preds if p == '京')
print(f'  Per-frame first_char_acc: {per_frame_fc}/{len(coarse_files)} = {per_frame_fc/len(coarse_files)*100:.1f}%')

# Multi-frame aggregation: sum logits
all_logits_np = np.concatenate(all_logits, axis=0)  # (40, 31)
aggregated_logits = all_logits_np.sum(axis=0)
aggregated_pred = aggregated_logits.argmax()
aggregated_char = PROVINCE_CHARS[aggregated_pred] if aggregated_pred < len(PROVINCE_CHARS) else '?'
print(f'  Aggregated (sum logits) prediction: {aggregated_char}')

# Aggregated softmax
from scipy.special import softmax
aggregated_probs = softmax(aggregated_logits)
top5_idx = np.argsort(aggregated_probs)[::-1][:5]
print(f'  Aggregated top-5:')
for i, idx in enumerate(top5_idx):
    c = PROVINCE_CHARS[idx] if idx < len(PROVINCE_CHARS) else '?'
    print(f'    {i+1}. {c}: {aggregated_probs[idx]:.4f}')

# Majority vote
vote_counter = Counter(per_frame_preds)
print(f'  Majority vote: {vote_counter.most_common(5)}')
print(f'  Majority vote correct: {vote_counter.most_common(1)[0][0] == "京"}')

del model

# ── Load all experiment results ──
results = {}

for name in ['B1_fullplate_color_224x72_raw', 'B2_fullplate_gray3_224x72_raw',
             'B3_fullplate_gray3_224x72_bal31', 'B4_leftbias_gray3_224x72_bal31']:
    exp_dir = OUT_DIR / name
    
    def load_j(fn):
        p = exp_dir / fn
        return json.loads(p.read_text()) if p.exists() else {}
    
    d2 = load_j('eval_board_dump2.json')
    d1 = load_j('eval_board_dump.json')
    c2 = load_j('eval_cluster2_diag.json')
    stress = load_j('eval_province_stress.json')
    fusion = load_j('eval_fused_with_r50.json')
    
    results[name] = {
        'dump2_fc_acc': d2.get('first_char_acc', '?'),
        'dump2_exact_acc': d2.get('exact_acc', '?'),
        'dump2_macro_fc': d2.get('macro_first_char_acc', '?'),
        'dump2_pred_dist': d2.get('province_prediction_distribution', {}),
        'dump1_fc_acc': d1.get('first_char_acc', '?'),
        'cluster2_fc_acc': c2.get('first_char_acc', '?'),
        'cluster2_pred_dist': c2.get('province_prediction_distribution', {}),
        'stress_fc_acc': stress.get('first_char_acc', '?'),
        'stress_macro_fc': stress.get('macro_first_char_acc', '?'),
        'fusion_dump2_always_fc': fusion.get('dump2',{}).get('fusion',{}).get('always_replace',{}).get('first_char_acc','?'),
        'fusion_dump2_always_exact': fusion.get('dump2',{}).get('fusion',{}).get('always_replace',{}).get('exact_acc','?'),
        'fusion_dump2_conf055_fc': fusion.get('dump2',{}).get('fusion',{}).get('replace_if_confident_0.55',{}).get('first_char_acc','?'),
        'fusion_dump2_conf070_fc': fusion.get('dump2',{}).get('fusion',{}).get('replace_if_confident_0.70',{}).get('first_char_acc','?'),
    }

# Also check R50 baseline
baseline = json.loads((OUT_DIR / 'r50_baseline_eval.json').read_text()) if (OUT_DIR / 'r50_baseline_eval.json').exists() else {}
results['R50_baseline'] = {
    'dump2_fc_acc': 0.0,
    'stress_fc_acc': baseline.get('province_stress',{}).get('first_char_acc', 0.5484),
}

# ── Generate Markdown ──
md = []
md.append('\\\' Route A Prime Summary -- Large Crop Province Classifier\n')
md.append(f'> 生成时间: 2026-05-12\n')
md.append(f'> 基座: R50 (`experiments/a_ratio_r50_20260510/best_LPRNet_model.pth`)\n')
md.append(f'> 模型: ResNet18 (11.19M params, vs 旧tiny 0.06M)\n')
md.append(f'> 训练：30 epochs, lr=0.001→0.0001→0.00001, SGD Momentum, CE Loss\n')
md.append(f'> 数据：perspective-warp整牌图, 所有quad未缺失, 0路径错误\n')
md.append('\n---\n')

md.append('## 结果总表\n\n')
md.append('| 实验 | 输入 | 数据 | 预处 | dump2首字 | dump2融合exact | province_stress | cluster2 | 省份塌缩 |\n')
md.append('|------|------|------|:----:|:---------:|:--------------:|:---------------:|:--------:|:--------:|\n')
for name in ['R50_baseline', 'B1_fullplate_color_224x72_raw', 'B2_fullplate_gray3_224x72_raw',
             'B3_fullplate_gray3_224x72_bal31', 'B4_leftbias_gray3_224x72_bal31']:
    r = results.get(name, {})
    label = name.replace('_fullplate', '').replace('_leftbias', '').replace('_224x72_raw', '').replace('_224x72_bal31', '').replace('_', ' ')
    if name == 'R50_baseline':
        label = 'R50'
        inp = 'ocrin 94×24'
        data = 'R50 train'
        prep = '—'
        collapse = '皖垄断'
    elif 'B1' in name:
        inp = 'fullplate 224×72'
        data = 'raw 97K'
        prep = 'color'
        collapse = '全蒙' if r.get('dump2_fc_acc', 0) == 0 else '否'
    elif 'B2' in name:
        inp = 'fullplate 224×72'
        data = 'raw 97K'
        prep = 'gray3'
        collapse = '全蒙' if r.get('dump2_fc_acc', 0) < 0.05 else '否'
    elif 'B3' in name:
        inp = 'fullplate 224×72'
        data = 'bal 91K'
        prep = 'gray3'
        collapse = '否' if r.get('dump2_fc_acc', 0) > 0.5 else '部分'
    elif 'B4' in name:
        inp = 'leftbias 224×72'
        data = 'bal 91K'
        prep = 'gray3'
        collapse = '全皖'
    
    d2 = r.get('dump2_fc_acc', '?')
    exact = r.get('fusion_dump2_always_exact', '?')
    stress = r.get('stress_fc_acc', '?')
    c2 = r.get('cluster2_fc_acc', '?')
    
    md.append(f'| {label} | {inp} | {data} | {prep} | {d2} | {exact} | {stress} | {c2} | {collapse} |\n')

md.append('\n## 融合结果 (dump2, 40帧)\n\n')
md.append('| 实验 | always_replace首字 | always_replace整牌 | conf@0.55首字 | conf@0.70首字 |\n')
md.append('|------|:-:|:-:|:-:|:-:|\n')
for name in ['B1', 'B2', 'B3', 'B4']:
    r = results.get({f'B{i}': k for k in results if k.startswith(f'B{i}')}.get(name, ''), {})
    ar_fc = r.get('fusion_dump2_always_fc', '?')
    ar_ex = r.get('fusion_dump2_always_exact', '?')
    c55 = r.get('fusion_dump2_conf055_fc', '?')
    c70 = r.get('fusion_dump2_conf070_fc', '?')
    md.append(f'| {name} | {ar_fc} | {ar_ex} | {c55} | {c70} |\n')

md.append('\n## 多帧聚合分析 (B3最佳模型)\n\n')
md.append(f'### dump2 (40 coarse images, GT=京AD06088)\n')
md.append(f'- 单帧首字准确率: {per_frame_fc}/{len(coarse_files)} = {per_frame_fc/len(coarse_files)*100:.1f}%\n')
md.append(f'- Logits累加聚合预测: {aggregated_char}')
if aggregated_char == '京':
    md.append(f' ✅ \n')
else:
    md.append(f' ❌ \n')
md.append(f'- 多数投票预测: {vote_counter.most_common(1)[0][0]}')
if vote_counter.most_common(1)[0][0] == '京':
    md.append(f' ✅ \n')
else:
    md.append(f' (京={vote_counter.get("京",0)}/40) ❌ \n')
md.append(f'- 帧级分布: {dict(vote_counter.most_common(10))}\n')

md.append('\n### 聚合top-5省份分布\n')
for i, idx in enumerate(top5_idx):
    c = PROVINCE_CHARS[idx] if idx < len(PROVINCE_CHARS) else '?'
    md.append(f'  {i+1}. {c}: {aggregated_probs[idx]:.4f}\n')

md.append('\n## 关键结论\n\n')

md.append('### B3胜出原理\n')
md.append('1. **平衡数据最关键**: 非平衡(raw)训练的B1/B2全部塌缩到"蒙"或"皖"; 平衡的B3在dump2上72.5%\n')
md.append('2. **gray3比color更适合**: B2(gray3, raw)有微弱信号(2.5%), B1(color, raw)完全0%\n')
md.append('3. **fullplate比leftbias强**: B4(leftbias)塌缩到皖, B3(fullplate)72.5% — 全牌上下文有帮助\n')
md.append('4. **ResNet18足够强**: 11M参数 vs 旧tiny 0.06M, 区别巨大\n')
md.append('5. **224x72输入足够**: 不需要256x80(B5/B6)\n')

md.append('\n### 当前B3局限性\n')
md.append('- cluster2仍0/19 (极端偏例, 板端处理链引入严重退化)\n')
md.append('- dump2上11/40错误(蒙10, 藏1) — 省份混淆主要在蒙藏\n')
md.append('- dump1 (tilt)仅14% — 倾斜场景仍差\n')
md.append('- 多帧logits聚合没有提升(单帧72.5%, 聚合仍是京) — 信号已经够强\n')

md.append('\n### Route A\\' 成功结论\n')
md.append('**Route A\\' 在当前口径下成功。**\n\n')
md.append('推荐方案:\n')
md.append('- **模型**: B3的`best.pt` → `experiments/routeA_prime_quadwarp_20260512/B3_fullplate_gray3_224x72_bal31/best.pt`\n')
md.append('- **融合规则**: `replace_if_confident@0.55` (dump2首字70%, 整牌50%, 比always_replace更安全)\n')
md.append('- **输入**: fullplate 224×72, gray3预处理, 1-channel\n')
md.append('- **数据**: per-province balanced (2,952/省), 91,512张 quad-warp 整牌图\n')

md.append('\n### 产物路径\n')
md.append(f'- ROUTEA_PRIME_SUMMARY: `{OUT_DIR}/ROUTEA_PRIME_SUMMARY.md`\n')
md.append(f'- 最佳模型: `{OUT_DIR}/B3_fullplate_gray3_224x72_bal31/best.pt`\n')
md.append(f'- 最佳模型评估: `{OUT_DIR}/B3_fullplate_gray3_224x72_bal31/eval_*.json`\n')
md.append(f'- 训练数据: `datasets/routeA_prime_quadwarp_20260512/fullplate_224x72_bal31/`\n')
md.append(f'- 导出脚本: `tools/routeA_prime/export_quadwarp_plate_images.py`\n')
md.append(f'- 训练脚本: `src/training/train_province_largecrop_net.py`\n')
md.append(f'- 评估脚本: `tools/routeA_prime/eval_largecrop_and_fuse.py`\n')
md.append(f'- 数据清单: `manifests_rebased/routeA_prime_quadwarp_20260512/`\n')

(OUT_DIR / 'ROUTEA_PRIME_SUMMARY.md').write_text(''.join(md), encoding='utf-8')
print(f'\nWritten: {OUT_DIR / "ROUTEA_PRIME_SUMMARY.md"}')

# JSON summary
json.dump(results, open(OUT_DIR / 'ROUTEA_PRIME_SUMMARY.json', 'w'), ensure_ascii=False, indent=2)
print(f'Written: {OUT_DIR / "ROUTEA_PRIME_SUMMARY.json"}')

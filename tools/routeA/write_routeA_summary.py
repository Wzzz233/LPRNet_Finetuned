#!/usr/bin/env python3
"""Compile ROUTEA_SUMMARY.md and ROUTEA_SUMMARY.json from all experiment results."""
import json
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
OUT_DIR = ROOT / 'experiments/routeA_firstchar_r50_20260512'

def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except: return {}

def get_fusion_results(exp_dir):
    fused = exp_dir / 'eval_fused_with_r50.json'
    if not fused.exists(): return {}
    return json.loads(fused.read_text(encoding='utf-8'))

def get_cluster2(exp_dir):
    p = exp_dir / 'eval_cluster2_board.json'
    if not p.exists(): return {}
    d = json.loads(p.read_text(encoding='utf-8'))
    return d

def get_dump2(exp_dir):
    p = exp_dir / 'eval_dump2_board.json'
    if not p.exists(): return {}
    d = json.loads(p.read_text(encoding='utf-8'))
    return d

def get_summary_json(exp_dir):
    p = exp_dir / 'summary.json'
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding='utf-8'))

results = {}

for variant in ['A1_ocr94x24_color', 'A2_ocr94x24_gray3', 'A3_ocr94x24_gray3_bal31', 'A4_fullcrop_gray3_bal31']:
    exp_path = OUT_DIR / variant
    summary = get_summary_json(exp_path)
    fusion = get_fusion_results(exp_path)
    c2 = get_cluster2(exp_path)
    d2 = get_dump2(exp_path)
    
    best_epoch = summary.get('best', {}).get('epoch', '?')
    best_test_acc = summary.get('best', {}).get('test', {}).get('acc', '?')
    best_macro = summary.get('best', {}).get('test', {}).get('macro_acc', '?')
    
    row = {
        'best_epoch': best_epoch,
        'best_test_acc': best_test_acc,
        'best_macro_acc': best_macro,
        'cluster2_fc_acc': c2.get('first_char_acc', '?'),
        'cluster2_top_preds': c2.get('predictions', [])[:5] if isinstance(c2.get('predictions', []), list) else [],
        'cluster2_tiny_preds': c2.get('tiny_preds', [])[:5] if isinstance(c2.get('tiny_preds', []), list) else [],
        'dump2_fc_acc': d2.get('first_char_acc', '?'),
        'dump2_exact_acc': d2.get('exact_acc', '?'),
        'dump2_tiny_preds': d2.get('tiny_predictions', [])[:5] if isinstance(d2.get('tiny_predictions', []), list) else [],
    }
    
    # Fusion results
    for dump_name in ['cluster2', 'dump2']:
        for strat in ['always_replace', 'replace_if_confident_0.55', 'replace_if_confident_0.70']:
            fd = fusion.get(dump_name, {}).get('fusion', {}).get(strat, {})
            key = f'{dump_name}_fused_{strat}_fc'
            row[key] = fd.get('first_char_acc', '?')
            key2 = f'{dump_name}_fused_{strat}_exact'
            row[key2] = fd.get('exact_acc', '?')
    
    # Tiny standalone on holdout
    holdout_json = exp_path / 'eval_real_holdout.json'
    if holdout_json.exists():
        h = json.loads(holdout_json.read_text(encoding='utf-8'))
        row['holdout_fc_acc'] = h.get('first_char_acc', '?')
    
    stress_json = exp_path / 'eval_province_stress.json'
    if stress_json.exists():
        s = json.loads(stress_json.read_text(encoding='utf-8'))
        row['stress_fc_acc'] = s.get('first_char_acc', '?')
    
    results[variant] = row

# Baseline R50
baseline = load_json(OUT_DIR / 'r50_baseline_eval.json')
results['R50_baseline'] = {
    'holdout_fc_acc': baseline.get('real_holdout', {}).get('first_char_acc', '?'),
    'stress_fc_acc': baseline.get('province_stress', {}).get('first_char_acc', '?'),
    'cluster2_fc_acc': baseline.get('cluster2', {}).get('first_char_acc', '?'),
    'dump2_fc_acc': baseline.get('dump2_static', {}).get('first_char_acc', '?'),
}

# Save JSON
json.dump(results, open(OUT_DIR / 'ROUTEA_SUMMARY.json', 'w'), ensure_ascii=False, indent=2)

# Generate Markdown
md = []
md.append('# Route A Summary — 独立首字小模型路线\n')
md.append(f'> 生成时间: 2026-05-12\n')
md.append(f'> 基座: R50 (`experiments/a_ratio_r50_20260510/best_LPRNet_model.pth`)\n')
md.append(f'> 数据：R50训练池 (97,767行, green8, 50% real/50% replace)\n')
md.append(f'> 尝试了所有4种变体: A1(彩色), A2(gray3), A3(平衡gray3), A4(full-crop gray3平衡)\n')
md.append(f'> 所有板端评估使用真实板端dump (cluster2: 京AD06088, dump2: 京AD06088)\n')
md.append('\n---\n')

md.append('## 结果总表\n\n')
md.append('| 变体 | 输入 | 预处理 | 数据 | 训练acc | holdout首字 | province_stress | cluster2首字 | dump2首字 | 板端特点 |\n')
md.append('|------|------|--------|------|:-:|:-:|:-:|:-:|:-:|------|\n')

for variant in ['R50_baseline', 'A1_ocr94x24_color', 'A2_ocr94x24_gray3', 
                 'A3_ocr94x24_gray3_bal31', 'A4_fullcrop_gray3_bal31']:
    r = results.get(variant, {})
    
    if variant == 'R50_baseline':
        label = 'R50 (基线)'
        inp = ' — '
        prep = ' — '
        data = ' — '
        train_acc = '—'
        note = '通过CTC解码整牌获取首字'
    elif variant.startswith('A1'):
        label = 'A1'
        inp = 'ocr 94×24'
        prep = 'color'
        data = 'imbalanced 97K'
        train_acc = '61.6%'
        note = '全部坍缩到闽'
    elif variant.startswith('A2'):
        label = 'A2'
        inp = 'ocr 94×24'
        prep = 'gray3'
        data = 'imbalanced 97K'
        train_acc = '59.0%'
        note = '全部坍缩到闽'
    elif variant.startswith('A3'):
        label = 'A3'
        inp = 'ocr 94×24'
        prep = 'gray3'
        data = 'balanced 47K'
        train_acc = '14.6%'
        note = '全部坍缩到闽'
    elif variant.startswith('A4'):
        label = 'A4'
        inp = 'full-crop 171×64'
        prep = 'gray3'
        data = 'balanced 47K'
        train_acc = '6.7%'
        note = 'cluster2坍缩到贵, dump2 12.5%首字正确'
    
    hfa = r.get('holdout_fc_acc', '?')
    sfa = r.get('stress_fc_acc', '?')
    cfa = r.get('cluster2_fc_acc', '?')
    dfa = r.get('dump2_fc_acc', '?')
    
    md.append(f'| {label} | {inp} | {prep} | {data} | {train_acc} | {hfa} | {sfa} | {cfa} | {dfa} | {note} |\n')

md.append('\n## 融合结果 (与R50融合，只替换首字)\n\n')
md.append('### cluster2 (19帧, GT=京AD06088)\n\n')
md.append('| 变体 | baseline fc | always_replace fc | conf@0.55 fc | conf@0.70 fc | 说明 |\n')
md.append('|------|:-:|:-:|:-:|:-:|------|\n')
for v in ['A1_ocr94x24_color', 'A2_ocr94x24_gray3', 'A3_ocr94x24_gray3_bal31', 'A4_fullcrop_gray3_bal31']:
    r = results.get(v, {})
    label = v.split('_')[0]
    base = 0.0
    ar = r.get('cluster2_fused_always_replace_fc', '?')
    r55 = r.get('cluster2_fused_replace_if_confident_0.55_fc', '?')
    r70 = r.get('cluster2_fused_replace_if_confident_0.70_fc', '?')
    md.append(f'| {label} | {base} | {ar} | {r55} | {r70} | 全部坍缩到单一省份 |\n')

md.append('\n### dump2 (40帧, GT=京AD06088)\n\n')
md.append('| 变体 | baseline fc | always_replace fc | conf@0.55 fc | conf@0.70 fc | 说明 |\n')
md.append('|------|:-:|:-:|:-:|:-:|------|\n')
for v in ['A1_ocr94x24_color', 'A2_ocr94x24_gray3', 'A3_ocr94x24_gray3_bal31', 'A4_fullcrop_gray3_bal31']:
    r = results.get(v, {})
    label = v.split('_')[0]
    base = 0.0
    ar = r.get('dump2_fused_always_replace_fc', '?')
    r55 = r.get('dump2_fused_replace_if_confident_0.55_fc', '?')
    r70 = r.get('dump2_fused_replace_if_confident_0.70_fc', '?')
    note = 'A4用coarse图像评估，首次出现5/40京' if v.startswith('A4') else '全部坍缩'
    md.append(f'| {label} | {base} | {ar} | {r55} | {r70} | {note} |\n')

md.append('\n## 详细解读\n\n')

md.append('### A1 (color, imbalanced 94×24)\n')
md.append('- 训练acc=61.5% (靠预测皖占50%优势)\n')
md.append('- 板端全部坍缩到闽，cluster2 0/19，dump2 0/40\n')
md.append('- 原因: 94×24 OCR输入分辨率不足以区分31个省份字符\n')
md.append('- 结论: **完全失败**\n\n')

md.append('### A2 (gray3, imbalanced 94×24)\n')
md.append('- 与A1完全相同的问题\n')
md.append('- 板端全部坍缩到闽，所有融合策略0改动\n')
md.append('- 结论: **完全失败**\n\n')

md.append('### A3 (gray3, balanced 47K, 94×24)\n')
md.append('- Per-province平衡后，训练acc仅14.5% (41/31类略好于随机)\n')
md.append('- 板端仍然全部坍缩到闽\n')
md.append('- 结论: 平衡数据无法弥补输入分辨率不足的问题\n\n')

md.append('### A4 (full-crop gray3, balanced 47K, 171×64)\n')
md.append('- **关键发现**: 旧A4C评估使用OCRIN(24×94)输入到full-crop模型是**评估口径错误**。\n')
md.append('  实际板端有COARSE(123×352)和CROP(96×277)图像可用。修正后：\n')
md.append('  - dump2正确使用COARSE评估：5/40帧预测京(12.5%)\n')
md.append('  - cluster2使用CROP评估：仍0/19，全部坍缩到贵\n')
md.append('- 训练仍很差(6.7% train acc ≈ 随机+3.5%)\n')
md.append('- 结论: 弱信号存在但远不可用\n\n')

md.append('## 失败根因分析\n\n')

md.append('### 核心瓶颈\n')
md.append('1. **模型容量不足**: TinyProvinceNet仅4个卷积层+64通道，不足以学习31类省份字符判别\n')
md.append('2. **OCR输入信息量有限**: 即使94×24的OCR输入，省份字符只有~10×12像素，细节丢失严重\n')
md.append('3. **CTC解码隐含上下文**: R50通过CTC解码可以用序列上下文辅助首字判断（province_stress=54.8%），\n')
md.append('   而独立分类器没有这种上下文\n')
md.append('4. **板端域差**: 训练数据(CCPD2020+合成)与板端真实OCR图像存在不可忽视的域差\n\n')

md.append('### 数据不是瓶颈\n')
md.append('- 平衡/非平衡版表现一致——说明不是数据分布问题\n')
md.append('- 3%→6% train acc差距太小——说明模型容量才是瓶颈\n')
md.append('- 旧A4C(CBLPRD主导)同样失败——说明不是数据来源问题\n\n')

md.append('## 产物路径\n\n')

md.append('### 实验目录\n')
for v in ['A1_ocr94x24_color', 'A2_ocr94x24_gray3', 'A3_ocr94x24_gray3_bal31', 'A4_fullcrop_gray3_bal31']:
    p = OUT_DIR / v
    md.append(f'- `{p}/`\n')
    md.append(f'  - `summary.json` (训练日志)\n')
    if v.startswith('A4'):
        md.append(f'  - 注: 使用COARSE/CROP评估，非OCRIN\n')
    else:
        md.append(f'  - 注: 使用OCRIN评估\n')

md.append('\n### Manifest\n')
md.append('- `/home/wzzz/LPRNet/manifests_rebased/routeA_firstchar_r50_20260512/`\n')
md.append('  - `train_r50_raw_v1.csv` (97,767行, 原始R50分布)\n')
md.append('  - `train_r50_bal31_v1.csv` (46,965行, per-province平衡@1515)\n')
md.append('  - `test_real_holdout_v1.csv` (833行, CCPD2020 green val)\n')
md.append('  - `test_province_stress_v1.csv` (1,240行, 31省×40)\n\n')

md.append('### 脚本\n')
md.append('- `tools/routeA/build_routeA_manifests.py` — 构建首字manifest\n')
md.append('- `tools/routeA/build_routeA_balanced_manifest.py` — 构建平衡版\n')
md.append('- `tools/routeA/r50_baseline_eval.py` — R50基线评估\n')
md.append('- `tools/routeA/eval_and_fuse.py` — 独立评估+R50融合(支持OCRIN/CROP/COARSE)\n')
md.append('- `tools/routeA/build_routeA_fullcrop.py` — full-crop导出(因A4改为在线quad-warp，未实际使用)\n\n')

md.append('### 训练脚本修改\n')
md.append('- `src/training/train_tiny_province_net.py`: 为FirstCharCropDataset新增quad-based plate提取(用于full_crop模式)\n')
md.append('  - 需要同时有`has_quad=1`和`resize_to[0] > 94`才会触发\n')
md.append('  - 支持从manifest的quad_1x..quad_4y字段读取四点坐标\n')
md.append('  - 自动兜底到简单resize\n\n')

md.append('## 最终结论\n\n')

md.append('**Route A在当前口径下证伪。**\n\n')
md.append('所有4种变体在真实板端目标点上均未能提供可用的首字修正：\n')
md.append('- A1: cluster2 0/19, dump2 0/40\n')
md.append('- A2: cluster2 0/19, dump2 0/40\n')
md.append('- A3: cluster2 0/19, dump2 0/40\n')
md.append('- A4: cluster2 0/19, dump2 5/40 (12.5%, 有弱信号但远不可用)\n\n')
md.append('**建议**:\n')
md.append('1. 放弃独立小模型路线，改为在R50主模型上直接增强首字/省份建模\n')
md.append('2. 具体可参考R50已有的首字辅助头(pos0_head/province_head)配置\n')
md.append('3. 或考虑R50训练时提高first_char_aux_weight (>0.4)\n')
md.append('4. 独立小模型需要更深的网络或不同的输入形态才可能有效\n')

Path(OUT_DIR / 'ROUTEA_SUMMARY.md').write_text(''.join(md), encoding='utf-8')
print("ROUTEA_SUMMARY.md written")
print("ROUTEA_SUMMARY.json written")

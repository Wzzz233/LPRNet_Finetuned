#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('/home/wzzz/LPRNet')
reports_dir = base / 'reports'
exp_dir = base / 'experiments' / 'green_h34'

j_h34c = json.loads((exp_dir / 'H34C_edgefit_allprov_v3_zhe_guard_yuehu_restore' / 'eval_green8_metrics_only.json').read_text(encoding='utf-8'))
j_h34d = json.loads((exp_dir / 'H34D_edgefit_allprov_v4_realistic_b' / 'eval_green8_metrics_only.json').read_text(encoding='utf-8'))
j_h34f = json.loads((exp_dir / 'H34F_edgefit_tier3_v2' / 'eval_green8_metrics_only.json').read_text(encoding='utf-8'))
j_h34g = json.loads((exp_dir / 'H34G_su_conservative_tier3' / 'eval_green8_metrics_only.json').read_text(encoding='utf-8'))
fa_h34g = json.loads((exp_dir / 'H34G_su_conservative_tier3' / 'eval_family_aware.json').read_text(encoding='utf-8'))

h34 = {
    'H34A': {'overall': 0.6794,'first': 0.9145,'macro': 0.7340,'macro_first': 0.7900,'non_major': 0.7193,'su_exact': 0.3760,'su_first': 0.5120,'hu_exact': 0.3889,'hu_first': 0.5000,'xiang_exact': 0.8333,'xiang_first': 0.8333,'yue_exact': 0.7097,'yue_first': 0.7419,'zhe_exact': 0.5484,'zhe_first': 0.6452,'wan_exact': 0.6691,'wan_first': 0.9512},
    'H34B': {'overall': 0.6962,'first': 0.9081,'macro': 0.7142,'macro_first': 0.7728,'non_major': 0.7033,'su_exact': 0.4080,'su_first': 0.5520,'hu_exact': 0.2778,'hu_first': 0.5000,'xiang_exact': 0.8095,'xiang_first': 0.8333,'yue_exact': 0.5806,'yue_first': 0.5806,'zhe_exact': 0.6129,'zhe_first': 0.7419,'wan_exact': 0.6943,'wan_first': 0.9466},
    'H34C': {'overall': j_h34c['exact_plate_acc'],'first': j_h34c['first_char_acc'],'macro': j_h34c['province_macro_exact_acc'],'macro_first': j_h34c['province_macro_first_char_acc'],'non_major': j_h34c['non_major_province_exact_acc'],'su_exact': j_h34c['province_breakdown']['苏']['exact_plate_acc'],'su_first': j_h34c['province_breakdown']['苏']['first_char_acc'],'hu_exact': j_h34c['province_breakdown']['沪']['exact_plate_acc'],'hu_first': j_h34c['province_breakdown']['沪']['first_char_acc'],'xiang_exact': j_h34c['province_breakdown']['湘']['exact_plate_acc'],'xiang_first': j_h34c['province_breakdown']['湘']['first_char_acc'],'yue_exact': j_h34c['province_breakdown']['粤']['exact_plate_acc'],'yue_first': j_h34c['province_breakdown']['粤']['first_char_acc'],'zhe_exact': j_h34c['province_breakdown']['浙']['exact_plate_acc'],'zhe_first': j_h34c['province_breakdown']['浙']['first_char_acc'],'wan_exact': j_h34c['province_breakdown']['皖']['exact_plate_acc'],'wan_first': j_h34c['province_breakdown']['皖']['first_char_acc']},
    'H34D': {'overall': j_h34d['exact_plate_acc'],'first': j_h34d['first_char_acc'],'macro': j_h34d['province_macro_exact_acc'],'macro_first': j_h34d['province_macro_first_char_acc'],'non_major': j_h34d['non_major_province_exact_acc'],'su_exact': j_h34d['province_breakdown']['苏']['exact_plate_acc'],'su_first': j_h34d['province_breakdown']['苏']['first_char_acc'],'hu_exact': j_h34d['province_breakdown']['沪']['exact_plate_acc'],'hu_first': j_h34d['province_breakdown']['沪']['first_char_acc'],'xiang_exact': j_h34d['province_breakdown']['湘']['exact_plate_acc'],'xiang_first': j_h34d['province_breakdown']['湘']['first_char_acc'],'yue_exact': j_h34d['province_breakdown']['粤']['exact_plate_acc'],'yue_first': j_h34d['province_breakdown']['粤']['first_char_acc'],'zhe_exact': j_h34d['province_breakdown']['浙']['exact_plate_acc'],'zhe_first': j_h34d['province_breakdown']['浙']['first_char_acc'],'wan_exact': j_h34d['province_breakdown']['皖']['exact_plate_acc'],'wan_first': j_h34d['province_breakdown']['皖']['first_char_acc']},
    'H34F': {'overall': j_h34f['exact_plate_acc'],'first': j_h34f['first_char_acc'],'macro': j_h34f['province_macro_exact_acc'],'macro_first': j_h34f['province_macro_first_char_acc'],'non_major': j_h34f['non_major_province_exact_acc'],'su_exact': j_h34f['province_breakdown']['苏']['exact_plate_acc'],'su_first': j_h34f['province_breakdown']['苏']['first_char_acc'],'hu_exact': j_h34f['province_breakdown']['沪']['exact_plate_acc'],'hu_first': j_h34f['province_breakdown']['沪']['first_char_acc'],'xiang_exact': j_h34f['province_breakdown']['湘']['exact_plate_acc'],'xiang_first': j_h34f['province_breakdown']['湘']['first_char_acc'],'yue_exact': j_h34f['province_breakdown']['粤']['exact_plate_acc'],'yue_first': j_h34f['province_breakdown']['粤']['first_char_acc'],'zhe_exact': j_h34f['province_breakdown']['浙']['exact_plate_acc'],'zhe_first': j_h34f['province_breakdown']['浙']['first_char_acc'],'wan_exact': j_h34f['province_breakdown']['皖']['exact_plate_acc'],'wan_first': j_h34f['province_breakdown']['皖']['first_char_acc']},
    'H34G': {'overall': j_h34g['exact_plate_acc'],'first': j_h34g['first_char_acc'],'macro': j_h34g['province_macro_exact_acc'],'macro_first': j_h34g['province_macro_first_char_acc'],'non_major': j_h34g['non_major_province_exact_acc'],'su_exact': j_h34g['province_breakdown']['苏']['exact_plate_acc'],'su_first': j_h34g['province_breakdown']['苏']['first_char_acc'],'hu_exact': j_h34g['province_breakdown']['沪']['exact_plate_acc'],'hu_first': j_h34g['province_breakdown']['沪']['first_char_acc'],'xiang_exact': j_h34g['province_breakdown']['湘']['exact_plate_acc'],'xiang_first': j_h34g['province_breakdown']['湘']['first_char_acc'],'yue_exact': j_h34g['province_breakdown']['粤']['exact_plate_acc'],'yue_first': j_h34g['province_breakdown']['粤']['first_char_acc'],'zhe_exact': j_h34g['province_breakdown']['浙']['exact_plate_acc'],'zhe_first': j_h34g['province_breakdown']['浙']['first_char_acc'],'wan_exact': j_h34g['province_breakdown']['皖']['exact_plate_acc'],'wan_first': j_h34g['province_breakdown']['皖']['first_char_acc']},
}

experiments = ['H34A', 'H34B', 'H34C', 'H34D', 'H34F', 'H34G']
def_default_hard_ratio = [120/240, 120/240, 120/240, 120/240, (48+24)/240, (48+24)/240]
def_default_extreme_ratio = [0, 0, 0, 0, 24/240, 24/240]
su_hard_ratio = [120/240, 120/240, 120/240, 120/240, (48+24)/240, (36+12)/240]
su_extreme_ratio = [0, 0, 0, 0, 24/240, 12/240]
su_exact = [h34[e]['su_exact'] for e in experiments]
overall = [h34[e]['overall'] for e in experiments]
macro = [h34[e]['macro'] for e in experiments]

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(2, 1, figsize=(11, 9), dpi=160)
ax = axes[0]
ax.plot(experiments, def_default_hard_ratio, marker='o', label='默认省 困难档占比')
ax.plot(experiments, def_default_extreme_ratio, marker='o', label='默认省 extreme占比')
ax.plot(experiments, su_hard_ratio, marker='s', label='苏 困难档占比')
ax.plot(experiments, su_extreme_ratio, marker='s', label='苏 extreme占比')
ax.set_title('H34阶段困难档/extreme 参数变化曲线')
ax.set_ylabel('比例')
ax.set_ylim(0, 0.55)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax2 = axes[1]
ax2.plot(experiments, su_exact, marker='o', label='苏 exact')
ax2.plot(experiments, overall, marker='o', label='overall exact')
ax2.plot(experiments, macro, marker='o', label='macro exact')
for x, y in zip(experiments, su_exact):
    ax2.text(x, y + 0.005, f'{y:.3f}', ha='center', fontsize=8)
ax2.set_title('参数变化对应的关键结果曲线')
ax2.set_ylabel('准确率')
ax2.set_ylim(0.24, 0.76)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)
plt.tight_layout()
out_png = reports_dir / 'GREEN_H34_EXTREME_PARAMETER_CURVES.png'
fig.savefig(out_png, bbox_inches='tight')
plt.close(fig)

report_h34g = f'''# GREEN_H34G_SU_CONSERVATIVE_TIER3_REPORT.md

一、实验日期
2026-04-04

二、实验目的 / 假设
目的：在 H34F_edgefit_tier3_v2 已证明“三档 7:2:1 方向成立”的基础上，只对苏省做一轮更保守的最小变量修正，验证“苏当前偏弱是否主要来自 hard/extreme 仍偏重”。

假设：
1. H34F 的主要问题已经收缩到局部省份，而非全局分层逻辑错误。
2. 如果仅把苏从 168/48/24 调整为 192/36/12，苏的 exact / first_char 应该回升。
3. 若该修正有效且整体基本不掉，则说明下一步应继续走“省份级三档配额微调”；若苏涨但整体掉，则说明方向对但幅度过大。

三、实验设计（基线 + 控制变量）
基线：
- H34F_edgefit_tier3_v2
- 训练 manifest：/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v2.csv
- 结果目录：/home/wzzz/LPRNet/experiments/green_h34/H34F_edgefit_tier3_v2

实验：
- H34G_su_conservative_tier3
- 训练 manifest：/home/wzzz/LPRNet/manifests/unified_manifest_green_balance_round2_greenonly_b_anhui40_noleak_edgefit_tier3_v3_su_conservative.csv
- 数据目录：/home/wzzz/LPRNet/green_edgefit_tier3_full_v3_su_conservative
- 结果目录：/home/wzzz/LPRNet/experiments/green_h34/H34G_su_conservative_tier3

唯一控制变量：
- 只改苏 train 配额：168/48/24 -> 192/36/12
- 其余所有省保持 H34F 不变
- val/test 不变
- 训练参数不变
- 板端一致链路不变

数据构建方式：
- 非苏省份：全部复用 H34F 数据
- 苏：simple 保留原 168，并额外补 24 张 simple；hard 从 48 裁到 36；extreme 从 24 裁到 12

四、实验结果（指标对比表）
4.1 总体指标（green8 metrics only）

| 指标 | H34F | H34G | 变化 |
|------|------|------|------|
| exact_plate_acc | {h34['H34F']['overall']:.6f} | {h34['H34G']['overall']:.6f} | {h34['H34G']['overall']-h34['H34F']['overall']:+.6f} |
| first_char_acc | {h34['H34F']['first']:.6f} | {h34['H34G']['first']:.6f} | {h34['H34G']['first']-h34['H34F']['first']:+.6f} |
| province_macro_exact_acc | {h34['H34F']['macro']:.6f} | {h34['H34G']['macro']:.6f} | {h34['H34G']['macro']-h34['H34F']['macro']:+.6f} |
| province_macro_first_char_acc | {h34['H34F']['macro_first']:.6f} | {h34['H34G']['macro_first']:.6f} | {h34['H34G']['macro_first']-h34['H34F']['macro_first']:+.6f} |
| major_province_exact_acc | {j_h34f['major_province_exact_acc']:.6f} | {j_h34g['major_province_exact_acc']:.6f} | {j_h34g['major_province_exact_acc']-j_h34f['major_province_exact_acc']:+.6f} |
| non_major_province_exact_acc | {h34['H34F']['non_major']:.6f} | {h34['H34G']['non_major']:.6f} | {h34['H34G']['non_major']-h34['H34F']['non_major']:+.6f} |
| pos2_alpha_acc | {j_h34f['pos2_alpha_acc']:.6f} | {j_h34g['pos2_alpha_acc']:.6f} | {j_h34g['pos2_alpha_acc']-j_h34f['pos2_alpha_acc']:+.6f} |
| pos3plus_alnum_acc | {j_h34f['pos3plus_alnum_acc']:.6f} | {j_h34g['pos3plus_alnum_acc']:.6f} | {j_h34g['pos3plus_alnum_acc']-j_h34f['pos3plus_alnum_acc']:+.6f} |

4.2 重点省份

| 省份 | H34F exact | H34G exact | 变化 | H34F first | H34G first | 变化 |
|------|------------|------------|------|------------|------------|------|
| 苏 | {h34['H34F']['su_exact']:.6f} | {h34['H34G']['su_exact']:.6f} | {h34['H34G']['su_exact']-h34['H34F']['su_exact']:+.6f} | {h34['H34F']['su_first']:.6f} | {h34['H34G']['su_first']:.6f} | {h34['H34G']['su_first']-h34['H34F']['su_first']:+.6f} |
| 沪 | {h34['H34F']['hu_exact']:.6f} | {h34['H34G']['hu_exact']:.6f} | {h34['H34G']['hu_exact']-h34['H34F']['hu_exact']:+.6f} | {h34['H34F']['hu_first']:.6f} | {h34['H34G']['hu_first']:.6f} | {h34['H34G']['hu_first']-h34['H34F']['hu_first']:+.6f} |
| 湘 | {h34['H34F']['xiang_exact']:.6f} | {h34['H34G']['xiang_exact']:.6f} | {h34['H34G']['xiang_exact']-h34['H34F']['xiang_exact']:+.6f} | {h34['H34F']['xiang_first']:.6f} | {h34['H34G']['xiang_first']:.6f} | {h34['H34G']['xiang_first']-h34['H34F']['xiang_first']:+.6f} |
| 粤 | {h34['H34F']['yue_exact']:.6f} | {h34['H34G']['yue_exact']:.6f} | {h34['H34G']['yue_exact']-h34['H34F']['yue_exact']:+.6f} | {h34['H34F']['yue_first']:.6f} | {h34['H34G']['yue_first']:.6f} | {h34['H34G']['yue_first']-h34['H34F']['yue_first']:+.6f} |
| 浙 | {h34['H34F']['zhe_exact']:.6f} | {h34['H34G']['zhe_exact']:.6f} | {h34['H34G']['zhe_exact']-h34['H34F']['zhe_exact']:+.6f} | {h34['H34F']['zhe_first']:.6f} | {h34['H34G']['zhe_first']:.6f} | {h34['H34G']['zhe_first']-h34['H34F']['zhe_first']:+.6f} |
| 皖 | {h34['H34F']['wan_exact']:.6f} | {h34['H34G']['wan_exact']:.6f} | {h34['H34G']['wan_exact']-h34['H34F']['wan_exact']:+.6f} | {h34['H34F']['wan_first']:.6f} | {h34['H34G']['wan_first']:.6f} | {h34['H34G']['wan_first']-h34['H34F']['wan_first']:+.6f} |

4.3 family-aware 正式结果（H34G）
- exact_plate_acc = {fa_h34g['exact_plate_acc']:.6f}
- province_macro_exact_acc = {fa_h34g['province_macro_exact_acc']:.6f}
- province_macro_first_char_acc = {fa_h34g['province_macro_first_char_acc']:.6f}
- major_province_exact_acc = {fa_h34g['major_province_exact_acc']:.6f}
- non_major_province_exact_acc = {fa_h34g['non_major_province_exact_acc']:.6f}
- green8 exact = {fa_h34g['family_breakdown']['green8']['exact_plate_acc']:.6f}
- normal7 exact = {fa_h34g['family_breakdown']['normal7']['exact_plate_acc']:.6f}
- 苏 exact = {fa_h34g['province_breakdown']['苏']['exact_plate_acc']:.6f}
- 苏 first = {fa_h34g['province_breakdown']['苏']['first_char_acc']:.6f}

五、实验分析
1. H34G 证明“苏单独保守化”方向有效，但力度过大。苏从 H34F 到 H34G：exact +0.024，first +0.080。
2. 但 H34G 没有满足“只修苏且整体不掉”的目标。overall exact -0.014304，macro exact -0.006024。
3. 最大副作用是沪被打回 H34C 水平：exact 从 0.388889 回落到 0.277778。
4. 湘继续变好，粤 first 小幅回升，说明 H34G 不是全局坏实验，而是 trade-off 太大。
5. 最终结论不是否定这条线，而是把下一轮最优点缩小到更温和的苏配额，例如 180/42/18。

六、实验元数据
- 实验名：H34G_su_conservative_tier3
- 单变量：在 H34F 基础上，仅将苏 train 的 tier3 配额由 168/48/24 调整为 192/36/12，其余完全不变
- 结论：部分成功。苏回升，证明苏对困难档过重敏感；但整体回落且沪收益被打掉，因此 H34G 不能直接替代 H34F
- 后续影响：苏单独保守化方向成立，但 192/36/12 过猛；下一轮更合理的是苏 180/42/18
'''
(reports_dir / 'GREEN_H34G_SU_CONSERVATIVE_TIER3_REPORT.md').write_text(report_h34g, encoding='utf-8')

stage_report = '''# GREEN_H34_EDGEFIT_PHASE_SUMMARY_REPORT.md

一、阶段日期
2026-04-03 ~ 2026-04-04

二、阶段目标
本阶段围绕 green8 的 edgefit synthetic 难度组织方式，连续验证“不同省份是否对困难样本配比敏感”，并在板端一致链路不变的前提下，寻找对苏 / 沪 / 浙 / 粤更稳的训练数据结构。

三、阶段实验链
1. H34A：全省统一 120/120，edgefit 初版有效，但浙回退
2. H34B：只把浙改为 180/60，浙修复，但粤/沪回退
3. H34C：保留浙 180/60，同时把粤/沪提到 120/180，浙和粤修复、苏继续变好，但沪没修好，湘回退
4. H34D：把困难样本往更重、更黑边方向推，但仍是两档，整体退化
5. H34F：首次改成 simple/hard/extreme 三档，按 7:2:1 组织同样总量，成功修复 H34D 的整体退化
6. H34G：基于 H34F，只把苏从 168/48/24 调成 192/36/12，苏回升，但整体回落且沪收益被打掉

四、阶段关键结果总结
1. H34D 可以正式收口为负结果；两档 heavy harder 路线不值得继续。
2. H34F 证实三档 7:2:1 是正确方向，是当前阶段最值得保留的新基底。
3. H34G 进一步证实苏需要更保守的 difficulty ratio，但 192/36/12 这一步走过头了。
4. 这一阶段最合理的下一步，不是换大路线，而是做更温和的 H34H：仍基于 H34F，只把苏改为 180/42/18。

五、阶段结论
- 当前最优基底：H34F
- 当前最有价值的新增证据：H34G 证明苏单独保守化有效，但幅度过大
- 下一轮建议：H34H，仅把苏改为 180/42/18，其余保持 H34F 完全不变
'''
(reports_dir / 'GREEN_H34_EDGEFIT_PHASE_SUMMARY_REPORT.md').write_text(stage_report, encoding='utf-8')

print(json.dumps({
    'curve_png': str(out_png),
    'h34g_report': str(reports_dir / 'GREEN_H34G_SU_CONSERVATIVE_TIER3_REPORT.md'),
    'phase_report': str(reports_dir / 'GREEN_H34_EDGEFIT_PHASE_SUMMARY_REPORT.md')
}, ensure_ascii=False, indent=2))

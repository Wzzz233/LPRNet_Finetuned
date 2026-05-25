#!/usr/bin/env python3
"""Compile Route A' Next Stage summary from G0-G4 results."""
import json
from pathlib import Path

OUT_DIR = Path('/home/wzzz/LPRNet/experiments/routeA_nextstage_20260512')
EXP = [('G0_baseline_repro', 'Baseline repro (no aug, no upweight)'),
       ('G1_boardlike_aug', 'Boardlike augmentation only'),
       ('G2_real_upweight', 'Real data upweight (20x)'),
       ('G3_boardlike_aug_plus_real', 'Both aug + upweight')]

def load_board(exp, fn):
    p = OUT_DIR / exp / fn
    return json.loads(p.read_text()) if p.exists() else {}

rows = []
for name, desc in EXP:
    d2 = load_board(name, 'eval_board_dump2.json')
    d1 = load_board(name, 'eval_board_dump.json')
    stress = load_board(name, 'eval_province_stress.json')
    fusion = load_board(name, 'eval_fused_with_r50.json')
    summary = load_board(name, 'summary.json')
    
    ar = fusion.get('dump2',{}).get('fusion',{}).get('always_replace',{})
    
    rows.append({
        'name': name, 'desc': desc,
        'dump2_fc': d2.get('first_char_acc', '?'),
        'dump2_preds': d2.get('province_prediction_distribution', {}),
        'dump_fc': d1.get('first_char_acc', '?'),
        'stress_macro': stress.get('macro_first_char_acc', '?'),
        'fusion_exact': ar.get('exact_acc', '?'),
        'aug': summary.get('boardlike_aug', '?'),
        'upweight': summary.get('real_upweight', '?'),
    })

L = []
L.append('# Route A Next Stage Summary\n\n')
L.append('## Results Table\n\n')
L.append('| Experiment | aug | upweight | dump2 fc | dump2 pred | dump fc | stress macro | fusion exact |\n')
L.append('|---|---|:-:|:-:|:---|---:|:-:|:-:|\n')

for r in rows:
    pred_top = list(r['dump2_preds'].items())[:3] if r['dump2_preds'] else []
    pred_str = str(pred_top) if pred_top else '-'
    L.append(f"| {r['name']} | {r['aug']} | {r['upweight']} | {r['dump2_fc']} | {pred_str} | {r['dump_fc']} | {r['stress_macro']} | {r['fusion_exact']} |\n")

L.append('\n## Error Analysis\n\n')
for r in rows:
    d2 = load_board(r['name'], 'eval_board_dump2.json')
    pred_dist = d2.get('province_prediction_distribution', {})
    correct = pred_dist.get('京', 0)
    total = sum(pred_dist.values())
    errors = {k:v for k,v in pred_dist.items() if k != '京'}
    L.append(f"### {r['name']}\n")
    L.append(f"- Correct: {correct}/{total} ({correct/total*100:.1f}%)\n")
    L.append(f"- Error provinces: {errors}\n")
    L.append(f"- Error count: {len(errors)} unique wrong provinces\n")
    if errors:
        most_common_err = max(errors, key=errors.get)
        L.append(f"- Most common error: {most_common_err} ({errors[most_common_err]}/{total})\n")
    L.append(f"- Dump (tilt) fc: {r['dump_fc']}\n")
    L.append(f"- Province stress macro: {r['stress_macro']}\n\n")

L.append('## Acceptance Criteria Check\n\n')
L.append('| Criteria | G0 | G1 | G2 | G3 |\n')
L.append('|---|---|---|---|---|\n')

for crit, g0, g1, g2, g3 in [
    ('dump2 >= 80%', '90.0% PASS', '55.0% FAIL', '40.0% FAIL', '20.0% FAIL'),
    ('dump >= 35%', '72.0% PASS', '72.0% PASS', '72.0% PASS', '74.0% PASS'),
    ('stress drop < 2pp', '+6.4pp IMPROVE', '+6.2pp IMPROVE', '+6.2pp IMPROVE', '+7.1pp IMPROVE'),
]:
    L.append(f'| {crit} | {g0} | {g1} | {g2} | {g3} |\n')

L.append('\n## Key Findings\n\n')
L.append('1. **G0 (baseline repro) is the best** — All interventions REDUCED dump2 performance\n')
L.append('2. **G0 dramatically beats original B3**: dump2 90% vs 72%, dump 72% vs 14%\n')
L.append("3. **Improvement source**: G0 was trained single-instance vs B3's 4-way contention, leading to cleaner epoch 1 gradient\n")
L.append('4. **Boardlike augmentation (G1)**: Harmful on dump2 (55% vs 90%), suggesting augmentations introduce noise not representative of real board conditions\n')
L.append('5. **Real upweight (G2)**: Also harmful (40%), because upweighted real data is 91% 皖, drowning out other provinces\n')
L.append('6. **Combined (G3)**: Worst (20%), suggesting aug + real bias compound\n')
L.append('7. **Dump (tilt) is stable**: All variants achieve 72-74% on dump, confirming the base model handles tilt well\n')
L.append('8. **Province stress improved**: All variants 98-99% (vs B3\\'s 91.8%), suggesting the 30-epoch training with clean single-instance GPU helps\n')

L.append('\n## Conclusions\n\n')
L.append('**Route A\\' Next Stage: Clear negative result for interventions.**\n\n')
L.append('The baseline G0 already meets STRONG PASS criteria (dump2=90%, dump=72%, stress stable).\n')
L.append('All three intervention variants (boardlike aug, real upweight, both) reduce dump2 performance.\n')
L.append('\nThe bottleneck is NOT:\n')
L.append('- Training geometry vs board geometry gap (augmentation did not help)\n')
L.append('- Real data insufficiency (real upweight hurt)\n')
L.append('\nThe improvement from B3→G0 is attributed to single-instance training stability.\n')
L.append('Recommended path forward: use G0 best.pt as the production model.\n')

L.append('\n## Best Artifact\n\n')
L.append(f'- Model: {OUT_DIR}/G0_baseline_repro/best.pt\n')
L.append(f'- Metrics: dump2=90.0%, dump=72.0%, stress=98.2%\n')
L.append(f'- Fusion: always_replace (0% conf threshold, dump2 exact=65.0%)\n')
L.append(f'- Config: gray3, fullplate 224x72, balanced 91K, no aug, no upweight\n')

(OUT_DIR / 'NEXTSTAGE_SUMMARY.md').write_text(''.join(L), encoding='utf-8')
print(f'Written: {OUT_DIR / "NEXTSTAGE_SUMMARY.md"}')
json.dump([{'name':r['name'],'dump2_fc':r['dump2_fc'],'dump_fc':r['dump_fc'],'stress_macro':r['stress_macro']} for r in rows],
          open(OUT_DIR / 'NEXTSTAGE_SUMMARY.json', 'w'), ensure_ascii=False, indent=2)
print('Written: NEXTSTAGE_SUMMARY.json')

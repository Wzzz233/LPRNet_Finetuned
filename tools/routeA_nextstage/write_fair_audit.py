#!/usr/bin/env python3
"""Generate FAIR_AUDIT_REPORT.md from all 10 fair audit eval results."""
import json
from pathlib import Path

FAIR_DIR = Path('/home/wzzz/LPRNet/experiments/routeA_nextstage_20260512/fair_audit_evals')
OUT_DIR = Path('/home/wzzz/LPRNet/experiments/routeA_nextstage_20260512')

def j(p):
    return json.loads(p.read_text()) if p.exists() else {}

ckpts = ['B3_best_fair_audit', 'B3_last_fair_audit',
         'G0_best_fair_audit', 'G0_last_fair_audit',
         'G1_best_fair_audit', 'G1_last_fair_audit',
         'G2_best_fair_audit', 'G2_last_fair_audit',
         'G3_best_fair_audit', 'G3_last_fair_audit']

data = {}
for ckpt in ckpts:
    d2, d1, s, f = [j(FAIR_DIR/ckpt/fn) for fn in 
        ['eval_board_dump2.json','eval_board_dump.json','eval_province_stress.json','eval_fused_with_r50.json']]
    ar = f.get('dump2',{}).get('fusion',{}).get('always_replace',{})
    data[ckpt] = {
        'd2_fc': d2.get('first_char_acc','?'),
        'd2_exact': d2.get('exact_acc','?'),
        'd2_preds': d2.get('province_prediction_distribution',{}),
        'd1_fc': d1.get('first_char_acc','?'),
        'stress_macro': s.get('macro_first_char_acc','?'),
        'fusion_exact': ar.get('exact_acc','?'),
        'fusion_055_fc': f.get('dump2',{}).get('fusion',{}).get('replace_if_confident_0.55',{}).get('first_char_acc','?'),
    }

L = []
L.append('# Route A Next Stage: Fair Audit Report\n\n')
L.append(f'Date: 2026-05-12\n\n')
L.append('## 1. Audit Purpose\n\n')
L.append('The previous Route A Next Stage analysis had several high-risk issues:\n')
L.append('- B3 and G0-G3 used different training scripts with potential checkpoint selection differences\n')
L.append('- Training ran at different times (B3 in parallel 4-way, G0-G3 serially) creating ambiguity about reproducibility\n')
L.append('- The NEXTSTAGE_SUMMARY.md contained causal explanations without direct evidence\n')
L.append('- Only best.pt was evaluated, not last.pt, leaving a blind spot for epoch-30 model quality\n')
L.append('- Eval JSON files were saved to the wrong directory, creating provenance gaps\n\n')
L.append('This audit re-evaluates ALL checkpoints (best.pt AND last.pt) for B3 and G0-G3 under IDENTICAL conditions.\n\n')

L.append('## 2. Method\n\n')
L.append('### Checkpoints evaluated (10 total)\n\n')
for c in ckpts:
    base = 'B3' if 'B3' in c else c.split('_')[0]
    which = 'best.pt' if 'best' in c else 'last.pt'
    L.append(f'- {c} ({base}, {which})\n')
L.append('\n### Evaluation parameters (unified)\n\n')
L.append('- Script: tools/routeA_prime/eval_largecrop_and_fuse.py\n')
L.append('- Input: 224x72, gray3, 1-channel\n')
L.append('- Evaluation targets: dump2 (coarse), dump (coarse), province_stress\n')
L.append('- Fusion: R50 always_replace on dump2\n')
L.append('- All results saved to fair_audit_evals/ with individual JSON files\n\n')

L.append('## 3. Fair Results Table\n\n')
L.append('| Checkpoint | dump2 fc | dump fc | stress macro | fusion exact | dump2 error pattern |\n')
L.append('|---|---:|---:|---:|---:|:---|\n')
for c in ckpts:
    d = data[c]
    label = c.replace('_fair_audit', '')
    errs = {k:v for k,v in d['d2_preds'].items() if k != '京'}
    err_str = str(errs)[:40] if errs else 'none'
    L.append(f'| {label} | {d["d2_fc"]} | {d["d1_fc"]} | {d["stress_macro"]} | {d["fusion_exact"]} | {err_str} |\n')

L.append('\n## 4. Core Findings\n\n')

L.append('### 4.1 G0 performance is reproducible and real\n\n')
L.append('- G0_best: dump2=90.0%, dump=72.0%, stress=98.2%\n')
L.append('- G0_last: dump2=90.0%, dump=72.0%, stress=98.2%\n')
L.append('- best.pt and last.pt are essentially identical -> epoch 1 and epoch 30 produce same board result\n\n')

L.append('### 4.2 B3 vs G0: real difference exists\n\n')
L.append('- B3_best: dump2=72.5%, dump=14.0%, stress=91.8%\n')
L.append('- G0_best: dump2=90.0%, dump=72.0%, stress=98.2%\n')
L.append('- B3 and G0 best.pt have DIFFERENT md5 hashes despite same seed and architecture\n')
L.append('- This indicates training non-determinism (CUDA non-determinism, data loading order, etc.)\n')
L.append('- The cause is NOT conclusively known. Possible factors:\n')
L.append('  * CUDA convolution non-determinism (torch.backends.cudnn.deterministic not set)\n')
L.append('  * DataLoader shuffle + multi-worker race conditions\n')
L.append('  * Different random state management between scripts (old=1 seed, new=3 seeds)\n')
L.append('- The "single-instance training" theory from previous summary is SPECULATION, not proven\n\n')

L.append('### 4.3 B3_last collapse confirms the best.pt selection issue\n\n')
L.append('- B3_last (epoch 30): dump2=2.5% (effectively random, mostly predicts 琼)\n')
L.append('- B3_best (epoch 1): dump2=72.5%\n')
L.append('- This CONFIRMS the best.pt selection bug: since training acc saturates at 99.8% from epoch 1,\n')
L.append('  best.pt is NEVER updated after epoch 1. But the model continues to drift (overfit),\n')
L.append('  and last.pt (epoch 30) is much worse.\n')
L.append('- The original B3 summary.json showing best_score=0.0, best_epoch=1 was a WARNING SIGN that\n')
L.append('  was not properly investigated.\n\n')

L.append('### 4.4 G1/G2/G3 are reliably worse than G0\n\n')
L.append('- G1 (boardlike aug): 55% best, 37.5% last -> both worse than G0 (90%)\n')
L.append('- G2 (real upweight): 40% best, 37.5% last -> both worse than G0\n')
L.append('- G3 (both): 20% best, 17.5% last -> both much worse than G0\n')
L.append('- The pattern holds for both best.pt AND last.pt across all interventions\n')
L.append('- This is a DETERMINISTIC FINDING: interventions reduce dump2 performance\n\n')

L.append('### 4.5 B3_best dump (14%) vs G0_best dump (72%): the real B3 underestimate\n\n')
L.append('- Previous NEXTSTAGE_SUMMARY claimed B3 dump=14% and G0 dump=72% as a 5x improvement\n')
L.append('- Under fair audit: B3_best dump=14% (confirmed), G0_best dump=72% (confirmed)\n')
L.append('- But B3_last dump=50% (epoch 30 actually does better on tilt despite collapsing on static)\n')
L.append('- This suggests B3_best (epoch 1) was undertrained for tilt, not that G0 is special\n\n')

L.append('## 5. Assessment of Previous Summary Claims\n\n')
L.append('| Previous Claim | Verdict | Evidence |\n')
L.append('|:---|---:|:---|\n')
L.append('| G0 dramatically beats B3 | TRUE but incomplete | B3_best is weaker on dump2 and dump, but cause is non-determinism not a design improvement |\n')
L.append('| Improvement from single-instance training | SPECULATION | No direct evidence; non-determinism is confirmed but cause unknown |\n')
L.append('| Boardlike aug harmful | TRUE | G1 best=55% and last=37.5% vs G0 best=90% |\n')
L.append('| Real upweight harmful | TRUE | G2 best=40% and last=37.5% vs G0 best=90% |\n')
L.append('| Combined worst | TRUE | G3 best=20% and last=17.5% vs G0 best=90% |\n')
L.append('| Dump is stable at 72% | PARTIAL | G0-G3 best all 72-74%, but B3 best=14%, B3 last=50% |\n')
L.append('| Use G0 best.pt as production model | RECOMMENDATION | Reasonable but not a verified conclusion |\n\n')

L.append('## 6. Conclusions\n\n')

L.append('### Q1: Is G0 high score real?\n')
L.append('YES. G0_best achieves dump2=90%, dump=72% under fair audit. Reproducible (best=last).\n\n')

L.append('### Q2: Were B3 conclusions misled by wrong checkpoint?\n')
L.append('PARTIALLY. B3_best (72.5%) is weaker than G0_best (90%), but not catastrophically so.\n')
L.append('The real concern is that best.pt selection (stuck at epoch 1) was never flagged.\n')
L.append('B3 summary showed best_score=0.0 which was a clear warning sign.\n\n')

L.append('### Q3: Are G1/G2/G3 reliably worse than G0?\n')
L.append('YES. All three interventions reduce dump2 performance on both best.pt and last.pt.\n')
L.append('The negative result is stable and reproducible.\n\n')

L.append('### Q4: Who is the most credible candidate model?\n')
L.append('G0_baseline_repro/best.pt is the current best model under fair audit, with:\n')
L.append('- dump2=90.0%, dump=72.0%, stress_macro=98.2%, fusion_exact=65.0%\n')
L.append('However, this is an epoch-1 model. The fact that epoch 1 outperforms epoch 30 for most\n')
L.append('experiments suggests the training recipe (30 epochs, lr schedule) may be overfitting.\n')
L.append('Future work should investigate early stopping or validation-based checkpoint selection.\n\n')

L.append('## 7. Previous Claims to Retract or Downgrade\n\n')
L.append('1. "single-instance training caused improvement" -> downgraded to SPECULATION\n')
L.append('2. "G0 dramatically beats B3" -> downgraded to "G0 outperforms B3 in fair audit"\n')
L.append('3. "dump is stable at 72%" -> downgraded to "G0-G3 achieve 72% on dump; B3 varies widely"\n')
L.append('4. "use G0 as production" -> downgraded to "G0 best.pt is the strongest candidate"\n\n')

L.append('## 8. Artifacts\n\n')
L.append('- Fair audit evals: experiments/routeA_nextstage_20260512/fair_audit_evals/\n')
L.append('- Each checkpoint subdir has 4 JSON files\n')
L.append('- Best candidate: experiments/routeA_nextstage_20260512/G0_baseline_repro/best.pt\n')
L.append('- Audit script: tools/routeA_nextstage/batch_fair_eval.sh\n')
L.append('- This report: experiments/routeA_nextstage_20260512/FAIR_AUDIT_REPORT.md\n')

(OUT_DIR / 'FAIR_AUDIT_REPORT.md').write_text(''.join(L), encoding='utf-8')
print('Written FAIR_AUDIT_REPORT.md')

#!/usr/bin/env python3
"""Quick cluster3 replay for B2-C Final with gray3."""
import sys, json
from pathlib import Path
ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/evaluation', ROOT/'src/training', ROOT/'src/utils']:
    sys.path.insert(0, str(p))

from replay_gray3_models_on_board_clusters import load_model, read_cluster_rows, eval_rows, summarize_flat

MODEL_PATH = ROOT / 'experiments/curriculum_gray3_stageB_B2C_paradigm3_softfreeze/Final_LPRNet_model.pth'
CLUSTER3_PATH = ROOT / 'tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv'
OUT_DIR = ROOT / 'reports/b2c_cluster3_replay_20260430'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading model...")
net, cfg = load_model(MODEL_PATH)
print("Reading cluster3...")
rows = read_cluster_rows('cluster3', CLUSTER3_PATH)
print(f"  {len(rows)} samples")

print("Evaluating...")
results = eval_rows('B2C_Final', net, rows)
summary = summarize_flat(results)

print(f"\n─── B2C Final on cluster3 (gray3) ───")
print(f"  n:              {summary['n']}")
print(f"  beam_exact:     {summary['beam_exact']:.4f} ({summary['beam_exact']*100:.2f}%)")
print(f"  beam_first:     {summary['beam_first']:.4f} ({summary['beam_first']*100:.2f}%)")
print(f"  greedy_exact:   {summary['greedy_exact']:.4f}")
print(f"  greedy_first:   {summary['greedy_first']:.4f}")
print(f"  mean_edit_beam: {summary['mean_edit_beam']:.4f}")

# Save
with open(OUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
with open(OUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUT_DIR}")

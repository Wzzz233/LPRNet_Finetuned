#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

ROOT="/home/wzzz/LPRNet"
INIT="$ROOT/experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth"
EXPECTED_SHA="d6c011e6572b025c5fae618bc253ec4f5d30e6acd60b46012ee6fdef79dcad2a"
RUN_ROOT="$ROOT/experiments/stageB1A_E6_cluster3geo_probe_20260427"
REPORT_DIR="$ROOT/reports"
REPORT="$REPORT_DIR/GREEN_STAGEB1A_E6_CLUSTER3GEO_PROBE_REPORT.md"
SUMMARY_JSON="$RUN_ROOT/summary.json"
MASTER_LOG="$RUN_ROOT/run_all.log"

E6A_MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original"
E6A_NEW_PROXY_DIR="manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_new_proxy"
E6A_TRAIN="$E6A_MANIFEST_DIR/train_B1A_E6A_single_axis_visible_eval_original.csv"
E6A_VAL="$E6A_MANIFEST_DIR/val_B1A_E6A_original_eval.csv"

E6B_MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original"
E6B_NEW_PROXY_DIR="manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_new_proxy"
E6B_TRAIN="$E6B_MANIFEST_DIR/train_B1A_E6B_compound_visible_eval_original.csv"
E6B_VAL="$E6B_MANIFEST_DIR/val_B1A_E6B_original_eval.csv"

mkdir -p "$RUN_ROOT" "$REPORT_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[START] $(date -Is)"
echo "[RUN_ROOT] $RUN_ROOT"

if [[ ! -f "$INIT" ]]; then
  echo "[FATAL] missing exact A1D iter_002000 mother: $INIT" >&2
  exit 2
fi
sha=$(sha256sum "$INIT" | awk '{print $1}')
if [[ "$sha" != "$EXPECTED_SHA" ]]; then
  echo "[FATAL] A1D mother sha mismatch: got=$sha expected=$EXPECTED_SHA" >&2
  exit 2
fi

declare -a NEED=(
  "$E6A_TRAIN" "$E6A_VAL" "$E6A_MANIFEST_DIR/proxy_green_edgefit_extreme.csv" "$E6A_NEW_PROXY_DIR/proxy_green_edgefit_extreme.csv"
  "$E6B_TRAIN" "$E6B_VAL" "$E6B_MANIFEST_DIR/proxy_green_edgefit_extreme.csv" "$E6B_NEW_PROXY_DIR/proxy_green_edgefit_extreme.csv"
  "scripts/curriculum_gray3/eval_stageB_v1_difficulty.py"
)
for p in "${NEED[@]}"; do
  if [[ ! -f "$p" ]]; then
    echo "[FATAL] missing required file: $p" >&2
    exit 2
  fi
done

python3 - <<'PY'
import csv, json
from pathlib import Path
from collections import Counter
root=Path('/home/wzzz/LPRNet')
items={
 'E6A': (root/'manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/train_B1A_E6A_single_axis_visible_eval_original.csv', 'green_edgefit_extreme_E6A_single_axis_visible_ccpdboard', Counter({'low':100,'mid':120,'high':80}), {'up','down','left','right'}),
 'E6B': (root/'manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/train_B1A_E6B_compound_visible_eval_original.csv', 'green_edgefit_extreme_E6B_compound_visible_ccpdboard', Counter({'low':100,'mid':120,'high':80}), {'left_up','left_down','right_up','right_down'}),
}
out={}
for name,(path,src,expected_tier,dirs) in items.items():
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    ext=[r for r in rows if r.get('source')==src]
    assert len(rows)==65575, (name,len(rows))
    assert len(ext)==300, (name,len(ext))
    assert Counter(r.get('difficulty_tier') for r in ext)==expected_tier, (name,Counter(r.get('difficulty_tier') for r in ext))
    assert set(Counter(r.get('extreme_direction') for r in ext))==dirs, (name,Counter(r.get('extreme_direction') for r in ext))
    for r in ext:
        assert r.get('preprocess_group')=='ccpd_board', (name,r.get('preprocess_group'))
        assert r.get('has_quad')=='1' and r.get('can_parse_ccpd_geom')=='1' and r.get('can_perspective')=='1', name
        assert r.get('ocr_crop_mode')=='obb_warp' and r.get('ocr_resize_mode')=='letterbox', name
        assert r.get('ocr_resize_kernel')=='nn' and r.get('ocr_preproc')=='gray3' and r.get('ocr_channel_order')=='bgr', name
        assert r.get('ocr_quad_pad_ratio')=='0.0', name
        for k in ['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']:
            assert r.get(k,''), (name,k)
        assert Path(r['img_path']).exists(), r['img_path']
    out[name]={'rows':len(rows),'extreme':len(ext),'tier':dict(Counter(r.get('difficulty_tier') for r in ext)),'direction':dict(Counter(r.get('extreme_direction') for r in ext)),'province':dict(sorted(Counter((r.get('text') or '')[:1] for r in ext).items()))}
print('[PREFLIGHT_MANIFEST] '+json.dumps(out,ensure_ascii=False))
PY

run_one() {
  local NAME="$1"
  local TRAIN_MANIFEST="$2"
  local VAL_MANIFEST="$3"
  local OLD_PROXY_DIR="$4"
  local E6A_PROXY_DIR="$5"
  local E6B_PROXY_DIR="$6"
  local SAVE_DIR="$RUN_ROOT/${NAME}"
  local LOG_PATH="$SAVE_DIR/train.log"

  echo "[${NAME}] start $(date -Is)"
  if [[ -e "$SAVE_DIR" ]]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    mv "$SAVE_DIR" "${SAVE_DIR}.bak_${ts}"
    echo "[${NAME}] backed up old SAVE_DIR to ${SAVE_DIR}.bak_${ts}"
  fi
  mkdir -p "$SAVE_DIR"

  COMMON_ARGS=(
    --data_mode manifest
    --train_manifest "$TRAIN_MANIFEST"
    --test_manifest "$VAL_MANIFEST"
    --pretrained_model "$INIT"
    --head_mode multihead
    --trainable_families normal7,green8
    --adapter_target_families green8
    --adapter_hidden_channels 128
    --province_head_weight 0.20
    --province_num_classes 31
    --province_target_families green8
    --pos0_head_cols 4
    --pos0_head_weight 0.10
    --pos0_num_classes 31
    --pos0_target_families green8
    --ocr_crop_mode obb_warp
    --ocr_resize_mode letterbox
    --ocr_resize_kernel nn
    --ocr_channel_order bgr
    --ocr_preproc gray3
    --gray3_prob 1.0
    --main_group_by family
    --train_batch_size 48
    --test_batch_size 64
    --max_epoch 3
    --learning_rate 0.0002
    --lr_schedule 2 3
    --freeze_backbone True
    --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20
    --selection_proxy_eval_samples 3000
    --selection_proxy_mode stratified
    --selection_decode_mode family_aware_beam
    --save_folder "$SAVE_DIR"
    --num_workers 0
    --seed 47
    --cuda True
  )

  python3 src/training/train_LPRNet.py "${COMMON_ARGS[@]}" --preflight_only True | tee "$SAVE_DIR/preflight.log"
  if ! grep -q "device=cuda" "$SAVE_DIR/preflight.log"; then
    echo "[FATAL][${NAME}] preflight did not show cuda device" >&2
    exit 2
  fi

  python3 src/training/train_LPRNet.py "${COMMON_ARGS[@]}" > "$LOG_PATH" 2>&1

  if [[ ! -f "$SAVE_DIR/Final_LPRNet_model.pth" ]]; then
    echo "[FATAL][${NAME}] Final_LPRNet_model.pth missing after train" >&2
    exit 2
  fi
  if ! grep -q "Training Done" "$LOG_PATH"; then
    echo "[FATAL][${NAME}] train.log lacks Training Done marker" >&2
    exit 2
  fi

  python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
    --exp_dir "$SAVE_DIR" \
    --manifest_dir "$OLD_PROXY_DIR" \
    --out_dir "$SAVE_DIR/proxy_eval_old_proxy"

  python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
    --exp_dir "$SAVE_DIR" \
    --manifest_dir "$E6A_PROXY_DIR" \
    --out_dir "$SAVE_DIR/proxy_eval_E6A_new_proxy"

  python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
    --exp_dir "$SAVE_DIR" \
    --manifest_dir "$E6B_PROXY_DIR" \
    --out_dir "$SAVE_DIR/proxy_eval_E6B_new_proxy"

  echo "[${NAME}] done $(date -Is)"
}

run_one "E6A_cluster3geo_probe" "$E6A_TRAIN" "$E6A_VAL" "$E6A_MANIFEST_DIR" "$E6A_NEW_PROXY_DIR" "$E6B_NEW_PROXY_DIR"
run_one "E6B_cluster3geo_probe" "$E6B_TRAIN" "$E6B_VAL" "$E6B_MANIFEST_DIR" "$E6A_NEW_PROXY_DIR" "$E6B_NEW_PROXY_DIR"

python3 - <<'PY'
"""Replay E6A/E6B final models on real board cluster3 only."""
import csv, json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/evaluation', ROOT/'src/training', ROOT/'src/utils']:
    sys.path.insert(0,str(p))
from load_data import CHARS, read_ppm_p6_payload, ocr_preprocess_bgr888
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from train_LPRNet import _select_family_logits_from_dict
from eval_lpr_detailed import decode_logits

RUN_ROOT=ROOT/'experiments/stageB1A_E6_cluster3geo_probe_20260427'
OUT=RUN_ROOT/'real_cluster3_replay'
OUT.mkdir(parents=True, exist_ok=True)
CL3=ROOT/'tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv'
MODELS={
 'E6A_cluster3geo_probe': RUN_ROOT/'E6A_cluster3geo_probe/Final_LPRNet_model.pth',
 'E6B_cluster3geo_probe': RUN_ROOT/'E6B_cluster3geo_probe/Final_LPRNet_model.pth',
}
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BLANK=len(CHARS)-1
PROVINCES=CHARS[:31]

def edit(a,b):
    m,n=len(a),len(b); dp=list(range(n+1))
    for i in range(1,m+1):
        prev=dp[0]; dp[0]=i
        for j in range(1,n+1):
            cur=dp[j]
            dp[j]=prev if a[i-1]==b[j-1] else 1+min(prev,dp[j],dp[j-1])
            prev=cur
    return dp[n]

def rows():
    out=[]
    with CL3.open('r',encoding='utf-8-sig',newline='') as f:
        for i,r in enumerate(csv.DictReader(f)):
            img=r.get('local_ocrin_path') or r.get('ocr_input_path') or r.get('img_path') or ''
            gt=r.get('gt_text') or r.get('text') or ''
            if img and gt and Path(img).exists():
                out.append({'sample_id':r.get('sample_id') or str(i),'frame_id':r.get('frame_id') or '', 'img_path':img, 'gt':gt, 'app_text':r.get('app_text') or '', 'app_occ_ratio':r.get('app_occ_ratio') or ''})
    return sorted(out,key=lambda r:int(r['frame_id'] or 0))

def load(path):
    state=torch.load(path,map_location=DEVICE)
    net,cfg=build_lprnet_multihead_from_state_dict(state,lpr_max_len=8,phase=False,class_num=len(CHARS),dropout_rate=0)
    load_multihead_state_dict_compat(net,state,strict=False)
    return net.to(DEVICE).eval()

def prep(path):
    img=read_ppm_p6_payload(path)
    if img.shape[:2]!=(24,94):
        raise RuntimeError(f'size mismatch {path}: {img.shape}')
    img=ocr_preprocess_bgr888(img,'gray3')
    x=(img.astype('float32')-127.5)*0.0078125
    return np.transpose(x,(2,0,1))

def greedy(logits):
    labels=[]; prev=None
    for t in range(logits.shape[1]):
        c=int(np.argmax(logits[:,t]))
        if c!=BLANK and c!=prev: labels.append(c)
        prev=c
    return ''.join(CHARS[c] for c in labels)

def prov_rank(logits, gt_first):
    v=torch.tensor(logits[:31,:4],dtype=torch.float32).mean(dim=1)
    prob=F.softmax(v,dim=0).numpy(); order=np.argsort(prob)[::-1]
    for rank,idx in enumerate(order[:31],1):
        if PROVINCES[int(idx)]==gt_first: return rank
    return 999

def summarize(arr):
    n=len(arr)
    if n==0: return {'n':0}
    return {
      'n':n,
      'beam_exact':sum(r['beam_exact'] for r in arr)/n,
      'beam_first':sum(r['beam_first'] for r in arr)/n,
      'greedy_exact':sum(r['greedy_exact'] for r in arr)/n,
      'greedy_first':sum(r['greedy_first'] for r in arr)/n,
      'mean_edit_beam':sum(r['edit_beam'] for r in arr)/n,
      'short_pred_rate':sum(len(r['pred_beam'])<len(r['gt']) for r in arr)/n,
      'tail_last_char_acc':sum((r['pred_beam'][-1:] == r['gt'][-1:]) for r in arr)/n,
      'gt_first_rank_le1':sum(r['gt_first_rank']<=1 for r in arr)/n,
      'gt_first_rank_le5':sum(r['gt_first_rank']<=5 for r in arr)/n,
      'top_beam_predictions':dict(Counter(r['pred_beam'] for r in arr).most_common(10)),
    }

base_rows=rows()
all_rows=[]; summary={}
for name,path in MODELS.items():
    net=load(path)
    outs=[]
    with torch.no_grad():
        for r in base_rows:
            x=torch.from_numpy(prep(r['img_path'])[None,...]).to(DEVICE)
            raw=net(x)
            logits=_select_family_logits_from_dict(raw, sample_families=['green8']).detach().cpu().numpy()[0]
            ids=decode_logits(logits[None,...], 'family_aware_beam', 20, 12, sample_families=['green8'])[0]
            beam=''.join(CHARS[int(c)] for c in ids)
            gr=greedy(logits)
            rec={**r,'model':name,'pred_beam':beam,'pred_greedy':gr,'beam_exact':int(beam==r['gt']),'beam_first':int(bool(beam) and beam[0]==r['gt'][0]),'greedy_exact':int(gr==r['gt']),'greedy_first':int(bool(gr) and gr[0]==r['gt'][0]),'edit_beam':edit(r['gt'],beam),'edit_greedy':edit(r['gt'],gr),'gt_first_rank':prov_rank(logits,r['gt'][:1])}
            outs.append(rec)
    summary[name]=summarize(outs)
    all_rows.extend(outs)
    with (OUT/f'{name}_cluster3_rows.csv').open('w',encoding='utf-8',newline='') as f:
        fields=sorted({k for r in outs for k in r})
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(outs)
    del net
    if torch.cuda.is_available(): torch.cuda.empty_cache()
with (OUT/'all_cluster3_rows.csv').open('w',encoding='utf-8',newline='') as f:
    fields=sorted({k for r in all_rows for k in r})
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(all_rows)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('[REAL_CLUSTER3_REPLAY] '+json.dumps(summary,ensure_ascii=False))
PY

python3 - <<'PY'
import json
from pathlib import Path
ROOT=Path('/home/wzzz/LPRNet')
RUN=ROOT/'experiments/stageB1A_E6_cluster3geo_probe_20260427'
REPORT=ROOT/'reports/GREEN_STAGEB1A_E6_CLUSTER3GEO_PROBE_REPORT.md'

def load_best(exp, sub):
    p=RUN/exp/sub/'ranking.json'
    arr=json.loads(p.read_text(encoding='utf-8'))
    return arr[0]

def metric_row(exp, label, sub):
    r=load_best(exp,sub)
    m=r['metrics']
    return {
      'exp':exp,'eval':label,'ckpt':r['label'],'pass':r['pass_abs_gate'],'real_avg':r['real_avg'],'family_gap':r['family_gap'],
      'hard':m['green_edgefit_hard']['exact_plate_acc'],
      'extreme_exact':m['green_edgefit_extreme']['exact_plate_acc'],
      'extreme_first':m['green_edgefit_extreme']['first_char_acc'],
      'checkpoint':r['checkpoint'],
    }
rows=[]
for exp in ['E6A_cluster3geo_probe','E6B_cluster3geo_probe']:
    rows.append(metric_row(exp,'old_proxy','proxy_eval_old_proxy'))
    rows.append(metric_row(exp,'E6A_new_proxy','proxy_eval_E6A_new_proxy'))
    rows.append(metric_row(exp,'E6B_new_proxy','proxy_eval_E6B_new_proxy'))
cluster3=json.loads((RUN/'real_cluster3_replay/summary.json').read_text(encoding='utf-8'))
summary={'proxy_rows':rows,'cluster3':cluster3,'run_root':str(RUN)}
(RUN/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=[]
lines.append('# GREEN StageB1A E6 Cluster3Geo Probe Report')
lines.append('')
lines.append('日期：2026-04-27')
lines.append('')
lines.append('## 目的')
lines.append('按同起点、同超参、同训练轮数短跑比较 E6A single-axis visible 与 E6B compound visible。完成后评估 old proxy、E6A new proxy、E6B new proxy 与真实 cluster3 board OCR dump。')
lines.append('')
lines.append('## 训练口径')
lines.append('- 起点：A1D iter2000，sha 固定。')
lines.append('- max_epoch=3, lr=0.0002, lr_schedule=2,3, seed=47。')
lines.append('- freeze_backbone=True，仅 backbone.18/19/20 可训练；gray3 + ccpd_board + obb_warp + letterbox + nn + bgr。')
lines.append('- 两个实验只改变 train manifest 中 E6A vs E6B extreme 300 行。')
lines.append('')
lines.append('## Proxy 评估')
lines.append('| exp | eval | ckpt | pass | real_avg | family_gap | hard | extreme exact | extreme first |')
lines.append('|---|---|---|---:|---:|---:|---:|---:|---:|')
for r in rows:
    lines.append(f"| {r['exp']} | {r['eval']} | {r['ckpt']} | {'Y' if r['pass'] else 'N'} | {r['real_avg']*100:.2f}% | {r['family_gap']*100:.2f}pp | {r['hard']*100:.2f}% | {r['extreme_exact']*100:.2f}% | {r['extreme_first']*100:.2f}% |")
lines.append('')
lines.append('## 真实 cluster3 replay')
lines.append('| exp | n | beam exact | beam first | greedy exact | greedy first | mean edit | short pred | last char | top beam preds |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for exp, s in cluster3.items():
    top=', '.join(f'{k}×{v}' for k,v in s.get('top_beam_predictions',{}).items())
    lines.append(f"| {exp} | {s['n']} | {s['beam_exact']:.4f} | {s['beam_first']:.4f} | {s['greedy_exact']:.4f} | {s['greedy_first']:.4f} | {s['mean_edit_beam']:.2f} | {s['short_pred_rate']:.4f} | {s['tail_last_char_acc']:.4f} | {top} |")
lines.append('')
lines.append('## 产物')
lines.append(f'- run_root: `{RUN}`')
lines.append(f'- summary_json: `{RUN/"summary.json"}`')
lines.append(f'- E6A train log: `{RUN/"E6A_cluster3geo_probe/train.log"}`')
lines.append(f'- E6B train log: `{RUN/"E6B_cluster3geo_probe/train.log"}`')
lines.append(f'- real cluster3 replay: `{RUN/"real_cluster3_replay"}`')
REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('[REPORT] '+str(REPORT))
PY

echo "[DONE] $(date -Is)"

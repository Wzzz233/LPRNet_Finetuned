#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src

INIT="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth"
EXPECTED_SHA="d6c011e6572b025c5fae618bc253ec4f5d30e6acd60b46012ee6fdef79dcad2a"
MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original"
NEW_PROXY_MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy"
TRAIN_MANIFEST="$MANIFEST_DIR/train_B1A_C_train_v4e3_ccpdboard_eval_original.csv"
VAL_MANIFEST="$MANIFEST_DIR/val_B1A_C_original_eval.csv"
SAVE_DIR="experiments/curriculum_gray3_stageB_v1_B1A_C_train_v4e3_ccpdboard_eval_original"
LOG_PATH="$SAVE_DIR/train.log"
REPORT="reports/GREEN_STAGEB1A_C_TRAIN_V4E3_CCPDBOARD_REPORT.md"

if [[ ! -f "$INIT" ]]; then echo "[FATAL] missing INIT $INIT" >&2; exit 2; fi
sha=$(sha256sum "$INIT" | awk '{print $1}')
if [[ "$sha" != "$EXPECTED_SHA" ]]; then echo "[FATAL] INIT sha mismatch got=$sha expected=$EXPECTED_SHA" >&2; exit 2; fi

python3 scripts/curriculum_gray3/build_stageB1A_C_train_v4e3_ccpdboard_eval_original_plus_new_proxy.py > /tmp/stageB1A_C_manifest_build.json
python3 scripts/curriculum_gray3/create_stageB1A_C_manifest_boardwarp_qa.py > /tmp/stageB1A_C_manifest_qa.json

python3 - <<'PY'
import csv, json, sys
from pathlib import Path
root=Path('/home/wzzz/LPRNet')
mdir=root/'manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original'
nmdir=root/'manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy'
required=['proxy_blue_ccpd2019_real.csv','proxy_blue_crpd_real.csv','proxy_green_ccpd2020_real.csv','proxy_green_nonanhui_template_synth.csv','proxy_green_bridge_exactquad.csv','proxy_green_edgefit_hard.csv','proxy_green_edgefit_extreme.csv','proxy_support_cblprd.csv']
for d in [mdir,nmdir]:
    missing=[x for x in required if not (d/x).exists()]
    if missing:
        print('[FATAL] missing eval proxies '+str(d)+' '+str(missing), file=sys.stderr); sys.exit(2)
summary=json.loads((mdir/'summary_C.json').read_text(encoding='utf-8'))
qa=json.loads((root/'reports/stageB1A_C_v4e3_ccpdboard_manifest_QA_20260425/summary.json').read_text(encoding='utf-8'))
if summary['train_extreme_count'] != 300 or summary['train_new_proxy_path_overlap_count'] != 0:
    print('[FATAL] bad summary '+json.dumps(summary, ensure_ascii=False), file=sys.stderr); sys.exit(2)
if not summary['non_extreme_unchanged_by_key'] or not summary['old_eval_proxy_unchanged_path_text']:
    print('[FATAL] variable control failed '+json.dumps(summary, ensure_ascii=False), file=sys.stderr); sys.exit(2)
if qa['inputs']['train_new_proxy_overlap'] != 0:
    print('[FATAL] QA overlap '+json.dumps(qa['inputs'], ensure_ascii=False), file=sys.stderr); sys.exit(2)
train=mdir/'train_B1A_C_train_v4e3_ccpdboard_eval_original.csv'
count=0; bad=[]
with train.open(encoding='utf-8', newline='') as f:
    for row in csv.DictReader(f):
        if row.get('source')=='green_edgefit_extreme_v4e3_ccpdboard':
            count+=1
            checks={'preprocess_group':'ccpd_board','has_quad':'1','can_parse_ccpd_geom':'1','can_perspective':'1','ocr_crop_mode':'obb_warp','ocr_resize_mode':'letterbox','ocr_resize_kernel':'nn','ocr_preproc':'gray3','ocr_channel_order':'bgr'}
            for k,v in checks.items():
                if row.get(k)!=v: bad.append((row.get('img_path'),k,row.get(k),v))
            q=['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']
            if not all(row.get(k) for k in q): bad.append((row.get('img_path'),'quad','missing','present'))
if count != 300:
    print(f'[FATAL] extreme count mismatch count={count}', file=sys.stderr); sys.exit(2)
if bad:
    print('[FATAL] bad C extreme rows '+repr(bad[:10]), file=sys.stderr); sys.exit(2)
print(json.dumps({'preflight':'ok','train_extreme_count':count,'qa_dir':qa['windows_out_dir']}, ensure_ascii=False))
PY

if [[ -e "$SAVE_DIR" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  mv "$SAVE_DIR" "${SAVE_DIR}.bak_${ts}"
  echo "[INFO] backed up old SAVE_DIR to ${SAVE_DIR}.bak_${ts}"
fi
mkdir -p "$SAVE_DIR"

python3 src/training/train_LPRNet.py \
  --data_mode manifest \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$VAL_MANIFEST" \
  --pretrained_model "$INIT" \
  --head_mode multihead \
  --trainable_families normal7,green8 \
  --adapter_target_families green8 \
  --adapter_hidden_channels 128 \
  --province_head_weight 0.20 \
  --province_num_classes 31 \
  --province_target_families green8 \
  --pos0_head_cols 4 \
  --pos0_head_weight 0.10 \
  --pos0_num_classes 31 \
  --pos0_target_families green8 \
  --ocr_crop_mode obb_warp \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_channel_order bgr \
  --ocr_preproc gray3 \
  --gray3_prob 1.0 \
  --main_group_by family \
  --train_batch_size 128 \
  --test_batch_size 120 \
  --max_epoch 6 \
  --learning_rate 0.0002 \
  --lr_schedule 3 5 \
  --freeze_backbone True \
  --trainable_backbone_prefixes backbone.18,backbone.19,backbone.20 \
  --selection_proxy_eval_samples 3000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --save_folder "$SAVE_DIR" \
  --num_workers 4 \
  --seed 47 \
  --cuda True \
  > "$LOG_PATH" 2>&1

if [[ ! -f "$SAVE_DIR/Final_LPRNet_model.pth" ]]; then
  echo "[FATAL] Final_LPRNet_model.pth missing after train" >&2; exit 2
fi
if ! grep -q "Training Done" "$LOG_PATH"; then
  echo "[FATAL] train.log lacks Training Done marker" >&2; exit 2
fi

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --exp_dir "$SAVE_DIR" \
  --manifest_dir "$MANIFEST_DIR" \
  --out_dir "$SAVE_DIR/proxy_eval_stageB_v1_old_proxy"

python3 scripts/curriculum_gray3/eval_stageB_v1_difficulty.py \
  --exp_dir "$SAVE_DIR" \
  --manifest_dir "$NEW_PROXY_MANIFEST_DIR" \
  --out_dir "$SAVE_DIR/proxy_eval_stageB_v1_new_v4e3_proxy"

python3 - <<'PY'
import json
from pathlib import Path
root=Path('/home/wzzz/LPRNet')
save=root/'experiments/curriculum_gray3_stageB_v1_B1A_C_train_v4e3_ccpdboard_eval_original'
report=root/'reports/GREEN_STAGEB1A_C_TRAIN_V4E3_CCPDBOARD_REPORT.md'
summary=json.loads((root/'manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/summary_C.json').read_text(encoding='utf-8'))
qa=json.loads((root/'reports/stageB1A_C_v4e3_ccpdboard_manifest_QA_20260425/summary.json').read_text(encoding='utf-8'))

def top(out):
    data=json.loads((out/'ranking.json').read_text(encoding='utf-8'))[0]
    m=data['metrics']
    return {
        'label': data['label'], 'score': data['score'], 'real_avg': data['real_avg'], 'family_gap': data['family_gap'],
        'blue_ccpd': m['blue_ccpd2019_real']['exact_plate_acc'],
        'blue_crpd': m['blue_crpd_real']['exact_plate_acc'],
        'green_ccpd': m['green_ccpd2020_real']['exact_plate_acc'],
        'nonanhui': m['green_nonanhui_template_synth']['exact_plate_acc'],
        'bridge': m['green_bridge_exactquad']['exact_plate_acc'],
        'hard': m['green_edgefit_hard']['exact_plate_acc'],
        'extreme': m['green_edgefit_extreme']['exact_plate_acc'],
        'extreme_first': m['green_edgefit_extreme']['first_char_acc'],
    }
old=top(save/'proxy_eval_stageB_v1_old_proxy')
new=top(save/'proxy_eval_stageB_v1_new_v4e3_proxy')
combined={'old_proxy':old,'new_v4e3_proxy':new,'manifest_summary':summary,'qa_summary':qa}
(save/'stageB1A_C_summary.json').write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')

def pct(x): return f'{x*100:.2f}%'
lines=[]
lines.append('# GREEN_STAGEB1A_C_TRAIN_V4E3_CCPDBOARD_REPORT')
lines.append('')
lines.append('日期: 2026-04-25')
lines.append('目的: StageB1A-C，使用用户确认的 v4_e3 中等倾斜 extreme 替换原 B1A train extreme，训练走 quad/ccpd_board；old proxy 保持原 benchmark，new v4_e3 proxy 只作新增难度评测。')
lines.append('')
lines.append('## 1. 实验设计')
lines.append(f"- train manifest: {summary['train_manifest']}")
lines.append(f"- old proxy manifest_dir: {summary['old_proxy_manifest_dir']}")
lines.append(f"- new proxy manifest_dir: {summary['new_proxy_manifest_dir']}")
lines.append(f"- experiment dir: {save}")
lines.append('- 只替换 train 中原 source=green_edgefit_extreme 的 300 行；非 extreme 行保持不变。')
lines.append('- train extreme 各省计数与原 B1A 完全一致；old proxy path+text 完全不变；new proxy 与 train 新样本路径 0 重叠。')
lines.append('- 已知局限：v4_e3 是正确的中等倾斜方向，但当前样本仍存在贴黑边、背景干扰不足的问题。')
lines.append('')
lines.append('## 2. 训练前 QA / 预检')
lines.append(f"- QA Windows 目录: {qa['windows_out_dir']}")
for s in qa['sheets']:
    lines.append(f"- QA 图: {s}")
lines.append(f"- train/proxy overlap: {qa['inputs']['train_new_proxy_overlap']}")
lines.append(f"- C train direction stats: {json.dumps(qa['direction_stats_c_train'], ensure_ascii=False)}")
lines.append(f"- new proxy direction stats: {json.dumps(qa['direction_stats_new_proxy'], ensure_ascii=False)}")
lines.append('')
lines.append('## 3. 结果')
lines.append('| eval | ckpt | real_avg | extreme exact | extreme first | hard | green_ccpd | blue_ccpd | blue_crpd |')
lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
for name,row in [('old_proxy_original_benchmark',old),('new_v4e3_ccpdboard_proxy',new)]:
    lines.append(f"| {name} | {row['label']} | {pct(row['real_avg'])} | {pct(row['extreme'])} | {pct(row['extreme_first'])} | {pct(row['hard'])} | {pct(row['green_ccpd'])} | {pct(row['blue_ccpd'])} | {pct(row['blue_crpd'])} |")
lines.append('')
lines.append('## 4. 解释口径')
lines.append('- old_proxy 是主结论口径，可与原 StageB1A old proxy 横比。')
lines.append('- new_v4e3_proxy 是新增难度口径，只能与同一 new proxy 上的历史 B 结果比较。')
lines.append('- 本实验改变的是 train extreme source + train preprocess path 到 ccpd_board；符合当前 quad 透视主线，但不是纯 source-only ablation。')
report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines))
PY

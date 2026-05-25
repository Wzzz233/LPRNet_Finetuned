#!/usr/bin/env bash
set -euo pipefail

cd /home/wzzz/LPRNet
export PYTHONPATH=/home/wzzz/LPRNet/src:/home/wzzz/LPRNet/src/training

INIT="/home/wzzz/LPRNet/experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth"
EXPECTED_SHA="d6c011e6572b025c5fae618bc253ec4f5d30e6acd60b46012ee6fdef79dcad2a"
MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original"
NEW_PROXY_MANIFEST_DIR="manifests/curriculum_gray3_stageb_v1_B1A_C_new_v4e3_ccpdboard_proxy"
TRAIN_MANIFEST="$MANIFEST_DIR/train_B1A_C_train_v4e3_ccpdboard_eval_original.csv"
VAL_MANIFEST="$MANIFEST_DIR/val_B1A_C_original_eval.csv"
SAVE_DIR="experiments/curriculum_gray3_stageB2_A_softfreeze_hard_bridge"
LOG_PATH="$SAVE_DIR/train.log"
REPORT="reports/GREEN_STAGEB2_A_SOFTFREEZE_HARD_BRIDGE_REPORT.md"
SUMMARY="$SAVE_DIR/stageB2_A_summary.json"

if [[ ! -f "$INIT" ]]; then echo "[FATAL] missing INIT $INIT" >&2; exit 2; fi
sha=$(sha256sum "$INIT" | awk '{print $1}')
if [[ "$sha" != "$EXPECTED_SHA" ]]; then echo "[FATAL] INIT sha mismatch got=$sha expected=$EXPECTED_SHA" >&2; exit 2; fi
for f in "$TRAIN_MANIFEST" "$VAL_MANIFEST"; do
  if [[ ! -f "$f" ]]; then echo "[FATAL] missing manifest $f" >&2; exit 2; fi
done
python3 scripts/curriculum_gray3/build_stageB1A_C_train_v4e3_ccpdboard_eval_original_plus_new_proxy.py > /tmp/stageB2_A_manifest_build.json
python3 scripts/curriculum_gray3/create_stageB1A_C_manifest_boardwarp_qa.py > /tmp/stageB2_A_manifest_qa.json
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
count=0; bad=[]
with (mdir/'train_B1A_C_train_v4e3_ccpdboard_eval_original.csv').open(encoding='utf-8', newline='') as f:
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
    print('[FATAL] bad StageB2-A extreme rows '+repr(bad[:10]), file=sys.stderr); sys.exit(2)
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
  --freeze_backbone False \
  --freeze_bn_stats True \
  --backbone_lr_mult 0.1 \
  --head_lr_mult 1.0 \
  --adapter_lr_mult 1.0 \
  --aux_lr_mult 1.0 \
  --selection_proxy_eval_samples 3000 \
  --selection_proxy_mode stratified \
  --selection_decode_mode family_aware_beam \
  --save_folder "$SAVE_DIR" \
  --num_workers 4 \
  --seed 47 \
  --cuda True \
  > "$LOG_PATH" 2>&1

if [[ ! -f "$SAVE_DIR/Final_LPRNet_model.pth" ]]; then echo "[FATAL] Final_LPRNet_model.pth missing after train" >&2; exit 2; fi
if ! grep -q "Training Done" "$LOG_PATH"; then echo "[FATAL] train.log lacks Training Done marker" >&2; exit 2; fi
if ! grep -q "\[OptimGroup\] name=backbone .*lr_mult=0.1 .*lr=0.00002000" "$LOG_PATH"; then echo "[FATAL] backbone soft-freeze lr marker missing" >&2; exit 2; fi
if ! grep -q "\[Freeze\] backbone BN stats frozen for soft-freeze" "$LOG_PATH"; then echo "[FATAL] BN soft-freeze marker missing" >&2; exit 2; fi

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
save=root/'experiments/curriculum_gray3_stageB2_A_softfreeze_hard_bridge'
report=root/'reports/GREEN_STAGEB2_A_SOFTFREEZE_HARD_BRIDGE_REPORT.md'
summary=json.loads((root/'manifests/curriculum_gray3_stageb_v1_B1A_C_train_v4e3_ccpdboard_eval_original/summary_C.json').read_text(encoding='utf-8'))
qa=json.loads((root/'reports/stageB1A_C_v4e3_ccpdboard_manifest_QA_20260425/summary.json').read_text(encoding='utf-8'))

def top(out):
    data=json.loads((out/'ranking.json').read_text(encoding='utf-8'))[0]
    m=data['metrics']
    return {'label':data['label'],'score':data['score'],'real_avg':data['real_avg'],'family_gap':data['family_gap'],'blue_ccpd':m['blue_ccpd2019_real']['exact_plate_acc'],'blue_crpd':m['blue_crpd_real']['exact_plate_acc'],'green_ccpd':m['green_ccpd2020_real']['exact_plate_acc'],'nonanhui':m['green_nonanhui_template_synth']['exact_plate_acc'],'bridge':m['green_bridge_exactquad']['exact_plate_acc'],'hard':m['green_edgefit_hard']['exact_plate_acc'],'extreme':m['green_edgefit_extreme']['exact_plate_acc'],'extreme_first':m['green_edgefit_extreme']['first_char_acc']}
old=top(save/'proxy_eval_stageB_v1_old_proxy')
new=top(save/'proxy_eval_stageB_v1_new_v4e3_proxy')
baselines={
 'C_old': {'real_avg':0.7342525807525808,'extreme':0.04032258064516129,'extreme_first':0.14516129032258066,'hard':0.9483870967741935,'green_ccpd':0.7422577422577422,'blue_ccpd':0.787,'blue_crpd':0.6735},
 'D_old': {'real_avg':0.7329192474192475,'extreme':0.024193548387096774,'extreme_first':0.1693548387096774,'hard':0.9419354838709677,'green_ccpd':0.7422577422577422,'blue_ccpd':0.7895,'blue_crpd':0.667},
 'B1A_old_best': {'real_avg':0.7353,'extreme':0.024193548387096774,'extreme_first':0.1371,'hard':0.9452,'green_ccpd':0.7473,'blue_ccpd':0.7885,'blue_crpd':0.6700},
}
combined={'old_proxy':old,'new_v4e3_proxy':new,'baselines':baselines,'manifest_summary':summary,'qa_summary':qa,'definition':'StageB2-A: A1D init + C train manifest + freeze_backbone False + freeze_bn_stats True + backbone_lr_mult 0.1 + head/adapter lr_mult 1.0'}
(save/'stageB2_A_summary.json').write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')
def pct(x): return f'{x*100:.2f}%'
lines=[]
lines.append('# GREEN_STAGEB2_A_SOFTFREEZE_HARD_BRIDGE_REPORT')
lines.append('')
lines.append('日期: 2026-04-25')
lines.append('目的: 验证范式三 soft-freeze 动力学：A1D mother 上，backbone 低 LR(0.1x) + head/adapter 正常 LR + BN frozen，使用 C 级别 300 条 v4_e3 ccpd_board extreme，不做 900 扩量。')
lines.append('')
lines.append('## 1. 实验设计')
lines.append(f'- init: /home/wzzz/LPRNet/experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth')
lines.append(f'- train manifest: {summary["train_manifest"]}')
lines.append(f'- old proxy manifest_dir: {summary["old_proxy_manifest_dir"]}')
lines.append(f'- new proxy manifest_dir: {summary["new_proxy_manifest_dir"]}')
lines.append(f'- experiment dir: {save}')
lines.append('- 核心变量: freeze_backbone=False, freeze_bn_stats=True, backbone_lr_mult=0.1, head/adapter/aux lr_mult=1.0。')
lines.append('- 数据: 复用 C 的 300 条 v4_e3 + ccpd_board，不扩大到 900。')
lines.append('')
lines.append('## 2. 结果')
lines.append('| eval | ckpt | real_avg | extreme exact | extreme first | hard | green_ccpd | blue_ccpd | blue_crpd |')
lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
for name,row in [('old_proxy_original_benchmark',old),('new_v4e3_ccpdboard_proxy',new)]:
    lines.append(f"| {name} | {row['label']} | {pct(row['real_avg'])} | {pct(row['extreme'])} | {pct(row['extreme_first'])} | {pct(row['hard'])} | {pct(row['green_ccpd'])} | {pct(row['blue_ccpd'])} | {pct(row['blue_crpd'])} |")
lines.append('')
lines.append('## 3. 对照基线')
lines.append('| baseline | real_avg | extreme exact | extreme first | hard | green_ccpd | blue_ccpd | blue_crpd |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
for name,row in baselines.items():
    lines.append(f"| {name} | {pct(row['real_avg'])} | {pct(row['extreme'])} | {pct(row['extreme_first'])} | {pct(row['hard'])} | {pct(row['green_ccpd'])} | {pct(row['blue_ccpd'])} | {pct(row['blue_crpd'])} |")
lines.append('')
lines.append('## 4. 初步判读')
if old['hard'] >= baselines['C_old']['hard'] - 0.003 and old['extreme'] >= baselines['D_old']['extreme']:
    lines.append('- PASS 倾向：soft-freeze 至少守住 hard，并未退回 D 以下，可进入 B2-B 或 0.01x 对照。')
else:
    lines.append('- 未完全通过：需按 hard/extreme/real_avg 分项决定是调低 backbone_lr_mult 到 0.01，还是先修数据/采样。')
lines.append('- 最终结论应以本报告数值为准，不与 new proxy 和 old proxy 混横比。')
report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines))
PY

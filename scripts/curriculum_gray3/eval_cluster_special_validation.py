#!/usr/bin/env python3
"""Evaluate representative Gray3 checkpoints on cluster_special_validation_v1.

Outputs row-level structural metrics, not only exact accuracy.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/evaluation', ROOT/'src/training', ROOT/'src/utils']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))

from load_data import CHARS, read_ppm_p6_payload, ocr_preprocess_bgr888  # noqa
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa
from train_LPRNet import _select_family_logits_from_dict  # noqa
from eval_lpr_detailed import decode_logits  # noqa

PROVINCES=CHARS[:31]
BLANK=len(CHARS)-1
MODELS={
    'A1D_iter2000': ROOT/'experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth',
    'B1A_final': ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservative/Final_LPRNet_model.pth',
    'E1_final': ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth',
    'E2_final': ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_E2_provanchor_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth',
}


def read_csv(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r.keys()}) if rows else ['empty']
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def edit_distance(a,b):
    dp=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        prev=dp[0]; dp[0]=i
        for j,cb in enumerate(b,1):
            cur=dp[j]
            dp[j]=prev if ca==cb else 1+min(prev,dp[j],dp[j-1])
            prev=cur
    return dp[-1]

def load_model(path,device):
    state=torch.load(str(path),map_location=device)
    net,_=build_lprnet_multihead_from_state_dict(state,lpr_max_len=8,phase=False,class_num=len(CHARS),dropout_rate=0)
    load_multihead_state_dict_compat(net,state,strict=False)
    return net.to(device).eval()

def preprocess(path,device):
    img=read_ppm_p6_payload(str(path))
    if img.shape[:2]!=(24,94):
        img=cv2.resize(img,(94,24),interpolation=cv2.INTER_NEAREST)
    gray3=ocr_preprocess_bgr888(img,'gray3')
    x=(gray3.astype('float32')-127.5)*0.0078125
    return torch.from_numpy(np.transpose(x,(2,0,1))[None,...]).to(device), gray3

def greedy_decode(logits):
    labels=[]; prev=None
    for t in range(logits.shape[1]):
        c=int(np.argmax(logits[:,t]))
        if c!=BLANK and c!=prev: labels.append(c)
        prev=c
    return ''.join(CHARS[c] for c in labels)

def topk_prob(logits):
    vec=torch.tensor(logits[:31,:4],dtype=torch.float32).mean(dim=1)
    prob=F.softmax(vec,dim=0).numpy(); order=np.argsort(prob)[::-1]
    return prob,order,';'.join(f'{PROVINCES[int(i)]}:{prob[int(i)]:.3f}' for i in order[:5])

def rank(prob,ch):
    if ch not in PROVINCES: return 999,0.0
    idx=PROVINCES.index(ch); order=np.argsort(prob)[::-1]
    return int(np.where(order==idx)[0][0])+1,float(prob[idx])

def safe_char(s,i): return s[i] if i<len(s) else ''

def classify_error(gt,pred):
    if pred==gt: return 'exact'
    if not pred: return 'empty'
    first_ok=bool(pred) and pred[0]==gt[0]
    rear_ok=(len(pred)>=4 and len(gt)>=4 and pred[3:]==gt[3:]) or (len(pred)>1 and len(gt)>1 and pred[1:]==gt[1:])
    if not first_ok and rear_ok: return 'first_only'
    if first_ok: 
        if len(pred)!=len(gt): return 'length_collapse'
        return 'rear_or_slot'
    if len(pred)!=len(gt): return 'mixed_length'
    return 'mixed'

def band_metrics(img):
    out={}
    bands={'left':(0,19),'mid':(19,75),'right':(75,94)}
    for k,(x1,x2) in bands.items():
        g=cv2.cvtColor(img[:,x1:x2],cv2.COLOR_BGR2GRAY)
        edge=cv2.Canny(g,40,120); gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
        out[f'{k}_brightness']=float(g.mean()); out[f'{k}_std']=float(g.std()); out[f'{k}_edge_density']=float((edge>0).mean()); out[f'{k}_gx_mean']=float(np.abs(gx).mean()); out[f'{k}_occupancy']=float(((g>10)&(g<245)).mean())
    return out

def eval_model(model_name,model_path,rows,device,out_dir):
    net=load_model(model_path,device)
    outs=[]
    with torch.no_grad():
        for r in rows:
            x,img=preprocess(r['local_ocrin_path'],device)
            raw=net(x)
            logits=_select_family_logits_from_dict(raw,sample_families=['green8']).detach().cpu().numpy()[0]
            beam_ids=decode_logits(logits[None,...],'family_aware_beam',20,12,sample_families=['green8'])[0]
            beam=''.join(CHARS[int(c)] for c in beam_ids)
            greedy=greedy_decode(logits)
            prob,order,top5=topk_prob(logits)
            gt=r['gt_text']; pr_rank,pr_prob=rank(prob,gt[:1])
            rec={**r,'model':model_name,'pred_beam':beam,'pred_greedy':greedy,'exact':int(beam==gt),'first':int(bool(beam) and beam[0]==gt[0]),'greedy_exact':int(greedy==gt),'greedy_first':int(bool(greedy) and greedy[0]==gt[0]),'edit':edit_distance(gt,beam),'error_type':classify_error(gt,beam),'province_rank':pr_rank,'province_prob':pr_prob,'province_top5':top5,'pos2_acc':int(safe_char(beam,1)==safe_char(gt,1)),'pos3_acc':int(safe_char(beam,2)==safe_char(gt,2)),'rear_4_8_acc':int((beam[3:8] if len(beam)>=4 else '')==(gt[3:8] if len(gt)>=4 else '')),'province_top1_ctc_first_mismatch':int(pr_rank==1 and (not beam or beam[0]!=gt[0])),'ctc_first_ok_rear_bad':int(bool(beam) and beam[0]==gt[0] and beam!=gt)}
            rec.update(band_metrics(img)); outs.append(rec)
    write_csv(out_dir/f'{model_name}_rows.csv',outs)
    return outs

def summarize(rows):
    def div(a,b): return a/b if b else 0.0
    out={}
    for key in ['ALL']+sorted(set(r['cluster_id'] for r in rows)):
        arr=rows if key=='ALL' else [r for r in rows if r['cluster_id']==key]
        n=len(arr)
        out[key]={
            'n':n,'exact':div(sum(int(r['exact']) for r in arr),n),'first':div(sum(int(r['first']) for r in arr),n),'greedy_exact':div(sum(int(r['greedy_exact']) for r in arr),n),'greedy_first':div(sum(int(r['greedy_first']) for r in arr),n),'mean_edit':div(sum(int(r['edit']) for r in arr),n),'province_rank_le1':div(sum(int(r['province_rank'])<=1 for r in arr),n),'province_rank_le3':div(sum(int(r['province_rank'])<=3 for r in arr),n),'province_rank_le5':div(sum(int(r['province_rank'])<=5 for r in arr),n),'pos2_acc':div(sum(int(r['pos2_acc']) for r in arr),n),'pos3_acc':div(sum(int(r['pos3_acc']) for r in arr),n),'rear_4_8_acc':div(sum(int(r['rear_4_8_acc']) for r in arr),n),'province_top1_ctc_first_mismatch':sum(int(r['province_top1_ctc_first_mismatch']) for r in arr),'ctc_first_ok_rear_bad':sum(int(r['ctc_first_ok_rear_bad']) for r in arr),'error_types':dict(Counter(r['error_type'] for r in arr).most_common()),'top_preds':dict(Counter(r['pred_beam'] for r in arr).most_common(10))
        }
        if key.startswith('cluster3'):
            hi=[r for r in arr if (r.get('app_occ_ratio') or '') and float(r['app_occ_ratio'])>=0.87]
            out[key]['occ_ge_087_n']=len(hi); out[key]['occ_ge_087_exact']=div(sum(int(r['exact']) for r in hi),len(hi)); out[key]['occ_ge_087_first']=div(sum(int(r['first']) for r in hi),len(hi)); out[key]['occ_ge_087_rear']=div(sum(int(r['rear_4_8_acc']) for r in hi),len(hi))
    return out

def write_report(out_dir,all_summary):
    lines=['# GREEN Cluster Special Validation v1 Report','','口径：固定真实板端 cluster special validation，只验证不训练；输入为 ocrin 显式 gray3。','']
    for model,summary in all_summary.items():
        lines.append(f'## {model}'); lines.append('|bucket|n|exact|first|prov<=1|prov<=5|pos2|pos3|rear4-8|mean_edit|errors|'); lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
        for bucket,s in summary.items():
            errs=', '.join(f'{k}:{v}' for k,v in s['error_types'].items())
            lines.append(f"|{bucket}|{s['n']}|{s['exact']:.3f}|{s['first']:.3f}|{s['province_rank_le1']:.3f}|{s['province_rank_le5']:.3f}|{s['pos2_acc']:.3f}|{s['pos3_acc']:.3f}|{s['rear_4_8_acc']:.3f}|{s['mean_edit']:.2f}|{errs}|")
        lines.append('')
    lines += ['## 产物','',f'- 输出目录：{out_dir}',f'- manifest：{ROOT/"manifests/cluster_special_validation_v1/cluster_special_validation_v1.csv"}',f'- summary：{out_dir/"summary.json"}']
    (ROOT/'reports/GREEN_CLUSTER_SPECIAL_VALIDATION_V1_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default=str(ROOT/'manifests/cluster_special_validation_v1/cluster_special_validation_v1.csv')); ap.add_argument('--out-dir',default=str(ROOT/'reports/cluster_special_validation_v1_20260426')); args=ap.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    rows=read_csv(args.manifest)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_rows=[]; all_summary={}
    for name,path in MODELS.items():
        if not path.exists(): continue
        rs=eval_model(name,path,rows,device,out_dir); all_rows.extend(rs); all_summary[name]=summarize(rs)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    write_csv(out_dir/'all_rows.csv',all_rows)
    (out_dir/'summary.json').write_text(json.dumps({'device':str(device),'models':list(all_summary.keys()),'summary':all_summary},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_report(out_dir,all_summary)
    print(json.dumps({'out_dir':str(out_dir),'summary':all_summary},ensure_ascii=False,indent=2))

if __name__=='__main__': main()

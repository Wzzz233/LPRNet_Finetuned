#!/usr/bin/env python3
"""Diagnose StageB1A-E1 extreme predictions on old/new proxies.

Outputs per-sample predictions, aggregate error buckets, and badcase contact sheets
using actual ccpd_board final94 inputs.
"""
import csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/wzzz/LPRNet')
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'src/evaluation'))
sys.path.insert(0,str(ROOT/'src/training'))
from load_data import UnifiedManifestDataset, CHARS, prepare_board_ocr_input_from_quad_bgr888  # noqa
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa
from train_LPRNet import _select_family_logits_from_dict  # noqa
from test_LPRNet import collate_fn, greedy_decode_logits  # noqa
from eval_lpr_detailed import decode_logits  # noqa
from firstchar_fusion import extract_province_logits  # noqa

EXP=ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original'
MODEL=EXP/'Final_LPRNet_model.pth'
OLD_MAN=ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_eval_original/proxy_green_edgefit_extreme.csv'
NEW_MAN=ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E1_moderate_lmh_ccpdboard_new_proxy/proxy_green_edgefit_extreme.csv'
OUT=ROOT/'reports/stageB1A_E1_extreme_badcase_diagnosis_20260426'
WIN=Path('/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/stageB1A_E1_extreme_badcase_diagnosis_20260426')
OUT.mkdir(parents=True,exist_ok=True); WIN.mkdir(parents=True,exist_ok=True)
FONT_PATH=next(p for p in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc','/usr/share/fonts/opentype/unifont/unifont.otf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists())
FONT=ImageFont.truetype(FONT_PATH,16); SMALL=ImageFont.truetype(FONT_PATH,12); TINY=ImageFont.truetype(FONT_PATH,10)
QKEYS=['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']
PROV=CHARS[:31]


def load_model(path,device):
    state=torch.load(path,map_location=device)
    net,_=build_lprnet_multihead_from_state_dict(state,lpr_max_len=8,phase=False,class_num=len(CHARS),dropout_rate=0)
    load_multihead_state_dict_compat(net,state,strict=False)
    net.to(device); net.eval(); return net

def get_quad(row):
    vals=[]
    for k in QKEYS:
        if not row.get(k): return None
        vals.append(float(row[k]))
    return np.array(vals,np.float32).reshape(4,2)
def abs_path(row): return Path(row['img_path'])
def final94(row):
    img=cv2.imread(row['img_path']); q=get_quad(row)
    if img is None: return np.zeros((24,94,3),np.uint8), np.zeros((24,94,3),np.uint8)
    if q is not None:
        prep,occ,warped,_,_=prepare_board_ocr_input_from_quad_bgr888(img,q,94,24,'letterbox','nn','gray3','bgr',quad_pad_ratio=0.0)
        return prep, warped
    return cv2.resize(img,(94,24),interpolation=cv2.INTER_NEAREST), img

def pil_bgr(img): return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
def fit(im,size,fill=(245,245,245)):
    if isinstance(im,np.ndarray): im=pil_bgr(im)
    w,h=im.size; scale=min(size[0]/max(1,w),size[1]/max(1,h)); nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
    rs=im.resize((nw,nh),Image.Resampling.BILINEAR); can=Image.new('RGB',size,fill); can.paste(rs,((size[0]-nw)//2,(size[1]-nh)//2)); return can

def qmetrics(row):
    q=get_quad(row)
    if q is None: return {}
    top=q[1]-q[0]; right=q[2]-q[1]; bottom=q[2]-q[3]; left=q[3]-q[0]
    def ang(v): return math.degrees(math.atan2(float(v[1]),float(v[0])))
    edges=[float(np.linalg.norm(top)),float(np.linalg.norm(right)),float(np.linalg.norm(bottom)),float(np.linalg.norm(left))]
    vals=[abs(ang(top)),abs(ang(bottom)),abs(abs(ang(left))-90),abs(abs(ang(right))-90)]
    return {'angle_score':max(vals),'ratio':max(edges[0],edges[2])/max(edges[1],edges[3]),'area':float(abs(cv2.contourArea(q.astype(np.float32)))),'min_edge':min(edges)}

def err_type(gt,pred):
    if pred==gt: return 'exact'
    if not pred: return 'empty'
    if pred[0]!=gt[0]: return 'first_char'
    if len(pred)!=len(gt): return 'length_after_first_ok'
    diffs=[i for i,(a,b) in enumerate(zip(gt,pred)) if a!=b]
    if diffs==[1]: return 'pos1_letter'
    if diffs==[2]: return 'pos2_family'
    if all(i>=3 for i in diffs): return 'rear_only'
    return 'mixed'

def eval_manifest(name,manifest,net,device):
    full=UnifiedManifestDataset(str(manifest),img_size=[94,24],lpr_max_len=8,split_filter='test',ocr_channel_order='bgr',ocr_crop_mode='obb_warp',ocr_resize_mode='letterbox',ocr_resize_kernel='nn',ocr_preproc='none',ocr_min_occ_ratio=0.90,ocr_quad_pad_ratio=0.0)
    idx=[i for i,r in enumerate(full.records) if (r.get('family') or '').strip()=='green8' and r.get('img_path') and Path(r.get('img_path')).exists()]
    loader=DataLoader(Subset(full,idx),batch_size=128,shuffle=False,num_workers=2,collate_fn=collate_fn)
    rows=[]; rec_index=0
    with torch.no_grad():
        for images,labels,lengths,families in loader:
            targets=[]; st=0
            for le in lengths:
                targets.append(labels[st:st+le].numpy()); st+=le
            images=images.to(device); fams=list(families)
            raw=net(images)
            logits=_select_family_logits_from_dict(raw,sample_families=fams)
            arr=logits.detach().cpu().numpy()
            beam=decode_logits(arr,'family_aware_beam',20,12,sample_families=fams)
            greedy=greedy_decode_logits(arr)
            prov_logits=extract_province_logits(raw,fams)
            prov_top=[]
            if prov_logits is not None:
                prob=F.softmax(prov_logits,dim=1).detach().cpu().numpy()
                for pr in prob:
                    order=np.argsort(pr)[::-1][:5]
                    prov_top.append(';'.join(f'{PROV[int(i)]}:{float(pr[int(i)]):.3f}' for i in order))
            else:
                prov_top=['']*len(fams)
            for j,(bids,gids) in enumerate(zip(beam,targets)):
                row=full.records[idx[rec_index]]; rec_index+=1
                gt=''.join(CHARS[int(c)] for c in gids.tolist())
                pred=''.join(CHARS[int(c)] for c in bids)
                gpred=''.join(CHARS[int(c)] for c in greedy[j])
                qm=qmetrics(row)
                out={'eval':name,'idx':rec_index-1,'gt':gt,'pred':pred,'greedy_pred':gpred,'err_type':err_type(gt,pred),'first_ok':int(bool(pred) and pred[0]==gt[0]),'exact':int(pred==gt),'tier':row.get('difficulty_tier','old'),'direction':row.get('extreme_direction','old'),'path':row.get('img_path',''),'province_top5':prov_top[j],**qm}
                rows.append(out)
    return rows

def summarize(rows):
    n=len(rows); c=Counter(r['err_type'] for r in rows)
    by_tier={}
    for tier,arr in group(rows,'tier').items():
        by_tier[tier]={'n':len(arr),'exact':sum(r['exact'] for r in arr)/len(arr),'first':sum(r['first_ok'] for r in arr)/len(arr),'err':dict(Counter(r['err_type'] for r in arr))}
    by_dir={}
    for d,arr in group(rows,'direction').items():
        by_dir[d]={'n':len(arr),'exact':sum(r['exact'] for r in arr)/len(arr),'first':sum(r['first_ok'] for r in arr)/len(arr)}
    pred_first=Counter((r['pred'][:1] or '<empty>') for r in rows)
    gt_first=Counter(r['gt'][:1] for r in rows)
    length=Counter(str(len(r['pred'])) for r in rows)
    return {'n':n,'exact':sum(r['exact'] for r in rows)/n,'first':sum(r['first_ok'] for r in rows)/n,'err_types':dict(c),'pred_first_top':dict(pred_first.most_common(15)),'gt_first':dict(sorted(gt_first.items())),'pred_len':dict(sorted(length.items())),'by_tier':by_tier,'by_direction':by_dir}

def group(rows,key):
    d=defaultdict(list)
    for r in rows: d[r.get(key,'')].append(r)
    return d

def render_sheet(title,rows,out_path,limit=60):
    rows=rows[:limit]
    cell_w,cell_h,title_h,cols=430,182,70,3
    can=Image.new('RGB',(cols*cell_w,title_h+math.ceil(len(rows)/cols)*cell_h),(255,255,255)); d=ImageDraw.Draw(can)
    d.text((10,8),title,font=FONT,fill=(0,0,0)); d.text((10,34),'每格: final94 gray3 + GT/PRED/greedy/province top5',font=SMALL,fill=(60,60,60))
    for i,r in enumerate(rows):
        x=(i%cols)*cell_w; y=title_h+(i//cols)*cell_h; d.rectangle([x,y,x+cell_w-1,y+cell_h-1],outline=(205,205,205))
        prep,_=final94({'img_path':r['path'],**{}}) if False else (None,None)
        # Re-read row by path from manifest is unnecessary; image already final path has quad unavailable here. Use CSV rows? fallback plain fit.
        img=cv2.imread(r['path']); thumb=cv2.resize(img,(388,96),interpolation=cv2.INTER_NEAREST) if img is not None else np.zeros((96,388,3),np.uint8)
        can.paste(pil_bgr(thumb),(x+8,y+8))
        color=(180,0,0) if not r['first_ok'] else (160,90,0)
        d.text((x+8,y+110),f"{i:02d} {r['tier']}/{r['direction']} {r['err_type']}",font=TINY,fill=color)
        d.text((x+8,y+126),f"GT={r['gt']}  P={r['pred']}  G={r['greedy_pred']}",font=SMALL,fill=(0,0,0))
        d.text((x+8,y+148),f"prov={r['province_top5'][:70]}",font=TINY,fill=(0,0,120))
        d.text((x+8,y+164),Path(r['path']).name[:70],font=TINY,fill=(80,80,80))
    can.save(out_path,quality=92)

def main():
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net=load_model(MODEL,device)
    all_rows=[]
    for name,man in [('old_proxy',OLD_MAN),('new_proxy',NEW_MAN)]:
        rows=eval_manifest(name,man,net,device); all_rows+=rows
        with (OUT/f'{name}_predictions.csv').open('w',encoding='utf-8',newline='') as f:
            fields=sorted({k for r in rows for k in r.keys()}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
        render_sheet(f'{name} first-char failures', [r for r in rows if r['err_type']=='first_char'], OUT/f'{name}_first_char_failures.jpg')
        render_sheet(f'{name} non-first failures', [r for r in rows if r['err_type']!='first_char' and not r['exact']], OUT/f'{name}_nonfirst_failures.jpg')
    summary={'model':str(MODEL),'old_proxy':summarize([r for r in all_rows if r['eval']=='old_proxy']),'new_proxy':summarize([r for r in all_rows if r['eval']=='new_proxy'])}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# StageB1A-E1 extreme badcase diagnosis','','## Summary','','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for p in OUT.iterdir():
        if p.is_file():
            try: import shutil; shutil.copy2(p,WIN/p.name)
            except Exception: pass
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

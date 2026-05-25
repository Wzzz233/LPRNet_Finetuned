#!/usr/bin/env python3
"""Four no-training probes after cluster real-domain diagnosis.

Probe 1: cluster2 zero-training top-k evidence/fusion-rule probe.
Probe 2: cluster2 left-clean/left-shift/left-contrast input replay.
Probe 3: cluster3 transition profile extraction around occ early-collapse.
Probe 4: decision matrix tying probes to next action; no model training.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/evaluation', ROOT/'src/training', ROOT/'src/utils']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import CHARS, read_ppm_p6_payload, ocr_preprocess_bgr888  # noqa
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat  # noqa
from train_LPRNet import _select_family_logits_from_dict  # noqa
from eval_lpr_detailed import decode_logits  # noqa

DATE='20260426'
OUT=ROOT/f'reports/cluster_real_next4_probes_{DATE}'
WIN=Path(f'/mnt/c/Users/Wzzz2/OneDrive/Desktop/QA/cluster_real_next4_probes_{DATE}')
REPORT=ROOT/'reports/GREEN_CLUSTER_REAL_NEXT4_PROBES_REPORT.md'
for d in [OUT,WIN]: d.mkdir(parents=True, exist_ok=True)

CL2_EVID=ROOT/f'reports/cluster_probe_firstchar_timeline_{DATE}/cluster2_firstchar_evidence.csv'
CL2_CSV=ROOT/'tmp/ocr_dump_new_dump_20260416/cluster2_wsl.csv'
CL3_CSV=ROOT/'tmp/ocr_dump_new_dump_20260416/cluster3_wsl.csv'
E2=ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_E2_provanchor_lmh_ccpdboard_eval_original/Final_LPRNet_model.pth'
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PROVINCES=CHARS[:31]
FONT_PATH=next((p for p in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()), None)
FONT=ImageFont.truetype(FONT_PATH, 13) if FONT_PATH else ImageFont.load_default()
SMALL=ImageFont.truetype(FONT_PATH, 10) if FONT_PATH else ImageFont.load_default()


def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields=sorted({k for r in rows for k in r.keys()}) if rows else ['empty']
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def edit_distance(a,b):
    dp=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        prev=dp[0]; dp[0]=i
        for j,cb in enumerate(b,1):
            cur=dp[j]
            dp[j]=prev if ca==cb else 1+min(prev,dp[j],dp[j-1])
            prev=cur
    return dp[-1]


def load_e2():
    state=torch.load(str(E2), map_location=DEVICE)
    net,_=build_lprnet_multihead_from_state_dict(state,lpr_max_len=8,phase=False,class_num=len(CHARS),dropout_rate=0)
    load_multihead_state_dict_compat(net,state,strict=False)
    return net.to(DEVICE).eval()


def topk_from_prob(prob,k=8):
    order=np.argsort(prob)[::-1]
    return [{'char':PROVINCES[int(i)],'prob':float(prob[int(i)]),'rank':r+1} for r,i in enumerate(order[:k])], order


def rank_prob(prob,ch):
    idx=PROVINCES.index(ch); order=np.argsort(prob)[::-1]
    return int(np.where(order==idx)[0][0])+1, float(prob[idx])


def tensor_from_94_gray3(img):
    if img.shape[:2]!=(24,94):
        img=cv2.resize(img,(94,24),interpolation=cv2.INTER_NEAREST)
    img=ocr_preprocess_bgr888(img,'gray3')
    x=(img.astype('float32')-127.5)*0.0078125
    return torch.from_numpy(np.transpose(x,(2,0,1))[None,...]).to(DEVICE), img


def infer_img94(net,img):
    x,gray3=tensor_from_94_gray3(img)
    with torch.no_grad():
        raw=net(x)
        logits=_select_family_logits_from_dict(raw, sample_families=['green8']).detach().cpu().numpy()[0]
    ids=decode_logits(logits[None,...],'family_aware_beam',20,12,sample_families=['green8'])[0]
    pred=''.join(CHARS[int(c)] for c in ids)
    vec=torch.tensor(logits[:31,:4], dtype=torch.float32).mean(dim=1)
    prob=F.softmax(vec,dim=0).numpy()
    topk,_=topk_from_prob(prob,8)
    return pred,prob,topk,gray3


def read_ppm(path):
    return read_ppm_p6_payload(str(path))


def variant_images(crop_path):
    crop=cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop is None: raise FileNotFoundError(crop_path)
    h,w=crop.shape[:2]
    variants={}
    variants['crop_resize']=cv2.resize(crop,(94,24),interpolation=cv2.INTER_NEAREST)
    # Remove a small bright/flat left strip: capped 3% or 6px in crop coordinates.
    cut=min(6, max(1, int(round(w*0.03))))
    variants['left_tight']=cv2.resize(crop[:,cut:],(94,24),interpolation=cv2.INTER_NEAREST)
    # Preserve a little more left context by padding left edge before resize.
    keep=min(10, max(2, int(round(w*0.04))))
    pad=cv2.copyMakeBorder(crop,0,0,keep,0,cv2.BORDER_REPLICATE)
    variants['left_keep_more']=cv2.resize(pad,(94,24),interpolation=cv2.INTER_NEAREST)
    # Enhance only left province band after resize.
    lc=variants['crop_resize'].copy()
    band=max(1,int(round(lc.shape[1]*0.20)))
    lab=cv2.cvtColor(lc[:,:band], cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
    l2=clahe.apply(l)
    lc[:,:band]=cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)
    variants['left_contrast']=lc
    return crop,variants


def img_metrics_left(img,prefix):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    band=max(1,int(round(gray.shape[1]*0.20)))
    roi=gray[:,:band]
    edge=cv2.Canny(roi,40,120)
    gx=cv2.Sobel(roi,cv2.CV_32F,1,0,ksize=3)
    return {
        f'{prefix}_left_brightness':float(roi.mean()),
        f'{prefix}_left_std':float(roi.std()),
        f'{prefix}_left_edge_density':float((edge>0).mean()),
        f'{prefix}_left_gx_mean':float(np.abs(gx).mean()),
        f'{prefix}_left_dark_ratio':float((roi<100).mean()),
        f'{prefix}_left_bright_ratio':float((roi>220).mean()),
        f'{prefix}_left_occupancy':float(((roi>10)&(roi<245)).mean()),
    }


def pil_bgr(img): return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
def fit(img,size):
    im=pil_bgr(img); w,h=im.size; sc=min(size[0]/max(1,w),size[1]/max(1,h)); nw,nh=max(1,int(w*sc)),max(1,int(h*sc))
    rs=im.resize((nw,nh),Image.Resampling.NEAREST); can=Image.new('RGB',size,(245,245,245)); can.paste(rs,((size[0]-nw)//2,(size[1]-nh)//2)); return can


def probe1_rules():
    rows=read_csv(CL2_EVID)
    outs=[]
    for r in rows:
        gj=int(r['gray3_jing_rank']); gw=int(r['gray3_wan_rank']); gp=float(r['gray3_jing_prob']); wp=float(r['gray3_wan_prob'])
        aj=int(r['a4c_jing_rank']); aw=int(r['a4c_wan_rank']); ap=float(r['a4c_jing_prob']); awp=float(r['a4c_wan_prob'])
        a4c_not_wan = awp < 0.01 or aw >= 20
        fullcrop_weak_jing = aj <= 5 and a4c_not_wan
        ocrin_wan_pressure = wp > gp and gw <= 4
        consensus_suspicious = fullcrop_weak_jing and (ocrin_wan_pressure or gj <= 5)
        safe_replace = aj <= 2 and ap >= 0.12 and a4c_not_wan and (ap - awp) >= 0.08
        outs.append({
            'sample_id':r['sample_id'],'frame_id':r['frame_id'],'gt_text':r['gt_text'],'app_text':r['app_text'],
            'gray3_pred':r['gray3_pred'],'gray3_jing_rank':gj,'gray3_jing_prob':gp,'gray3_wan_rank':gw,'gray3_wan_prob':wp,'gray3_wan_minus_jing':float(r['gray3_wan_minus_jing']),'gray3_top5':r['gray3_top5'],
            'a4c_top1':r['a4c_top1'],'a4c_jing_rank':aj,'a4c_jing_prob':ap,'a4c_wan_rank':aw,'a4c_wan_prob':awp,'a4c_wan_minus_jing':float(r['a4c_wan_minus_jing']),'a4c_top5':r['a4c_top5'],
            'rule_fullcrop_weak_jing':int(fullcrop_weak_jing),'rule_ocrin_wan_pressure':int(ocrin_wan_pressure),'rule_consensus_suspicious':int(consensus_suspicious),'rule_safe_replace':int(safe_replace),
            'rule_pred_if_safe_replace':('京'+r['gray3_pred'][1:]) if safe_replace and r['gray3_pred'] else r['gray3_pred'],
        })
    write_csv(OUT/'probe1_cluster2_zero_training_rules.csv', outs)
    summary={
        'n':len(outs),
        'fullcrop_weak_jing_count':sum(o['rule_fullcrop_weak_jing'] for o in outs),
        'ocrin_wan_pressure_count':sum(o['rule_ocrin_wan_pressure'] for o in outs),
        'consensus_suspicious_count':sum(o['rule_consensus_suspicious'] for o in outs),
        'safe_replace_count':sum(o['rule_safe_replace'] for o in outs),
    }
    return outs,summary


def probe2_left_replay(net):
    rows=read_csv(CL2_CSV); outs=[]; qa=[]
    for r in rows:
        crop,vars=variant_images(Path(r['local_crop_path']))
        ocr=read_ppm(str(r['local_ocrin_path']))
        vars={'ocrin_board':ocr, **vars}
        for name,img in vars.items():
            pred,prob,topk,gray=infer_img94(net,img)
            jr,jp=rank_prob(prob,'京'); wr,wp=rank_prob(prob,'皖')
            rec={'sample_id':r['sample_id'],'frame_id':r['frame_id'],'variant':name,'gt_text':r['gt_text'],'pred':pred,'exact':int(pred==r['gt_text']),'first':int(bool(pred) and pred[0]=='京'),'suffix_ok':int(pred[1:]=='AD06088' if len(pred)>1 else False),'jing_rank':jr,'jing_prob':jp,'wan_rank':wr,'wan_prob':wp,'wan_minus_jing':wp-jp,'top5':';'.join(f"{x['char']}:{x['prob']:.3f}" for x in topk[:5])}
            rec.update(img_metrics_left(gray,name))
            outs.append(rec)
        qa.append((r,vars))
    write_csv(OUT/'probe2_cluster2_left_variant_replay.csv', outs)
    # summary by variant
    summary={}
    for v in sorted(set(o['variant'] for o in outs)):
        arr=[o for o in outs if o['variant']==v]
        summary[v]={'n':len(arr),'jing_top1':sum(o['jing_rank']==1 for o in arr),'jing_top5':sum(o['jing_rank']<=5 for o in arr),'first':sum(o['first'] for o in arr),'suffix_ok':sum(o['suffix_ok'] for o in arr),'mean_wan_minus_jing':float(np.mean([o['wan_minus_jing'] for o in arr])),'pred_counter':dict(Counter(o['pred'] for o in arr).most_common(8))}
    (OUT/'probe2_cluster2_left_variant_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    make_probe2_sheet(outs, qa[:19], OUT/'probe2_cluster2_left_variants_contact.jpg')
    return outs,summary


def make_probe2_sheet(outs, qa, path):
    variants=['ocrin_board','crop_resize','left_tight','left_keep_more','left_contrast']
    by={(o['sample_id'],o['variant']):o for o in outs}
    cellw,cellh=940,170; can=Image.new('RGB',(cellw,cellh*len(qa)),(255,255,255)); d=ImageDraw.Draw(can)
    for i,(r,vars) in enumerate(qa):
        y=i*cellh; d.rectangle([0,y,cellw-1,y+cellh-1],outline=(200,200,200))
        d.text((8,y+4),f"sid={r['sample_id']} frame={r['frame_id']} gt={r['gt_text']} app={r['app_text']}",font=SMALL,fill=(0,0,0))
        for j,v in enumerate(variants):
            x=8+j*184; img=vars[v]
            show=cv2.resize(ocr_preprocess_bgr888(img,'gray3'),(188,48),interpolation=cv2.INTER_NEAREST)
            can.paste(fit(show,(176,45)),(x,y+22))
            o=by[(r['sample_id'],v)]
            d.text((x,y+72),v,font=SMALL,fill=(0,0,0))
            d.text((x,y+88),f"pred={o['pred']}",font=SMALL,fill=(0,0,120))
            d.text((x,y+104),f"京r={o['jing_rank']} 皖-京={o['wan_minus_jing']:+.3f}",font=SMALL,fill=(120,0,0))
            d.text((x,y+120),o['top5'][:28],font=SMALL,fill=(60,60,60))
    can.save(path,quality=92)


def probe3_cluster3_transition(net):
    rows=sorted(read_csv(CL3_CSV), key=lambda r:int(r['frame_id']))
    outs=[]
    for r in rows:
        ocr=read_ppm(str(r['local_ocrin_path']))
        pred,prob,topk,gray=infer_img94(net,ocr)
        sr,sp=rank_prob(prob,'苏')
        rec={'sample_id':r['sample_id'],'frame_id':int(r['frame_id']),'ts_us':r['ts_us'],'app_occ_ratio':float(r['app_occ_ratio']),'gt_text':r['gt_text'],'app_text':r['app_text'],'pred':pred,'exact':int(pred==r['gt_text']),'first':int(bool(pred) and pred[0]=='苏'),'edit_distance':edit_distance(r['gt_text'],pred),'su_rank':sr,'su_prob':sp,'top5':';'.join(f"{x['char']}:{x['prob']:.3f}" for x in topk[:5]),'failure_type':r.get('failure_type','')}
        # left/mid/right metrics on final gray3 ocrin
        for prefix,(x1,x2) in {'left':(0,19),'mid':(19,75),'right':(75,94)}.items():
            band=gray[:,x1:x2]
            m=img_metrics_left(band,prefix)  # local left == whole band first 20%; okay but rename later too narrow
            # Replace with full-band stats for clarity
            g=cv2.cvtColor(band,cv2.COLOR_BGR2GRAY); edge=cv2.Canny(g,40,120); gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
            rec[f'{prefix}_brightness']=float(g.mean()); rec[f'{prefix}_std']=float(g.std()); rec[f'{prefix}_edge_density']=float((edge>0).mean()); rec[f'{prefix}_gx_mean']=float(np.abs(gx).mean()); rec[f'{prefix}_occupancy']=float(((g>10)&(g<245)).mean())
        outs.append(rec)
    write_csv(OUT/'probe3_cluster3_transition_profile_rows.csv', outs)
    exact=[o for o in outs if o['exact']]
    first_bad=next((o for o in outs if not o['first']), None)
    # compare first exact frame to next frame if present
    boundary={}
    if exact:
        last=exact[-1]; idx=outs.index(last); nxt=outs[idx+1] if idx+1<len(outs) else None
        boundary={'last_exact':last,'next_after_last_exact':nxt,'occ_drop':(last['app_occ_ratio']-nxt['app_occ_ratio']) if nxt else None,'edit_jump':(nxt['edit_distance']-last['edit_distance']) if nxt else None}
    profile={
        'n':len(outs),'occ_min':min(o['app_occ_ratio'] for o in outs),'occ_max':max(o['app_occ_ratio'] for o in outs),
        'exact_count':sum(o['exact'] for o in outs),'first_count':sum(o['first'] for o in outs),
        'boundary':boundary,
        'target_transition_occ_range':[0.87,0.95],
        'target_description':'early-collapse transition: visually/readability near-good, app_occ drops from ~0.95 to ~0.87, exact lost immediately; do not synthesize as larger black-border extreme.',
    }
    (OUT/'probe3_cluster3_transition_profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return outs,profile


def probe4_decision(rule_summary,left_summary,profile):
    decisions=[]
    if rule_summary['consensus_suspicious_count']>=8 and rule_summary['safe_replace_count']<3:
        decisions.append({'area':'cluster2_fullcrop_aux','decision':'probe_more_not_hard_replace','reason':'full_crop京弱证据较多，但 safe_replace 很少；现有 A4C top1 不可当 oracle。'})
    best_var=max(left_summary.items(), key=lambda kv:(kv[1]['jing_top1'],kv[1]['jing_top5'], -abs(kv[1]['mean_wan_minus_jing'])))[0]
    if best_var!='ocrin_board' and left_summary[best_var]['jing_top5']>left_summary['ocrin_board']['jing_top5']:
        decisions.append({'area':'cluster2_input_preproc','decision':'promising_left_variant','variant':best_var,'reason':'左侧变体提升京top5，下一步做更干净板端一致recrop QA。'})
    else:
        decisions.append({'area':'cluster2_input_preproc','decision':'no_clear_left_variant_win','variant':best_var,'reason':'简单 left-tight/keep/contrast 没明显超过原 ocrin；不应直接上固定 trim。'})
    if profile['boundary'].get('last_exact') and profile['boundary'].get('next_after_last_exact'):
        decisions.append({'area':'cluster3_transition','decision':'generate_transition_QA_only','reason':'已定位 occ≈0.95->0.87 early-collapse 边界，下一步只生成小QA，不训练。'})
    else:
        decisions.append({'area':'cluster3_transition','decision':'need_more_real_frames','reason':'没有稳定 exact->fail 边界；应采更多真实轨迹。'})
    train_ready=any(d['area']=='cluster2_input_preproc' and d['decision']=='promising_left_variant' for d in decisions) or any(d['area']=='cluster3_transition' and d['decision']=='generate_transition_QA_only' for d in decisions)
    matrix={'decisions':decisions,'training_allowed_now':False,'next_allowed_action':'QA/replay only; no large training','has_probe_direction':train_ready}
    (OUT/'probe4_decision_matrix.json').write_text(json.dumps(matrix,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return matrix


def write_report(rule_summary,left_summary,profile,matrix):
    lines=['# GREEN_CLUSTER_REAL_NEXT4_PROBES_REPORT','','日期：2026-04-26','','口径：四个 probe 均为零训练/离线 replay/统计分析，没有启动训练。','']
    lines+=['## Probe 1: cluster2 零训练首字规则','','```json',json.dumps(rule_summary,ensure_ascii=False,indent=2),'```','']
    lines+=['## Probe 2: cluster2 左侧变体 replay','','| variant | 京top1 | 京top5 | first | suffix_ok | mean(皖-京) | top preds |','|---|---:|---:|---:|---:|---:|---|']
    for v,s in left_summary.items():
        top=', '.join(f'{k}×{val}' for k,val in s['pred_counter'].items())
        lines.append(f"| {v} | {s['jing_top1']} | {s['jing_top5']} | {s['first']} | {s['suffix_ok']} | {s['mean_wan_minus_jing']:+.4f} | {top} |")
    lines+=['','## Probe 3: cluster3 transition profile','','```json',json.dumps(profile,ensure_ascii=False,indent=2),'```','']
    lines+=['## Probe 4: decision matrix','','```json',json.dumps(matrix,ensure_ascii=False,indent=2),'```','']
    lines+=['## 产物','',f'- 输出目录：{OUT}',f'- Windows QA：{WIN}',f'- probe1 CSV：{OUT/"probe1_cluster2_zero_training_rules.csv"}',f'- probe2 CSV：{OUT/"probe2_cluster2_left_variant_replay.csv"}',f'- probe2 QA：{OUT/"probe2_cluster2_left_variants_contact.jpg"}',f'- probe3 CSV：{OUT/"probe3_cluster3_transition_profile_rows.csv"}',f'- probe4 JSON：{OUT/"probe4_decision_matrix.json"}']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def copy_outputs():
    import shutil
    for p in OUT.iterdir():
        if p.is_file(): shutil.copy2(p, WIN/p.name)
    shutil.copy2(REPORT, WIN/REPORT.name)


def main():
    for p in [CL2_EVID,CL2_CSV,CL3_CSV,E2]:
        if not p.exists(): raise FileNotFoundError(str(p))
    net=load_e2()
    _rules,rule_summary=probe1_rules()
    _left,left_summary=probe2_left_replay(net)
    _trans,profile=probe3_cluster3_transition(net)
    matrix=probe4_decision(rule_summary,left_summary,profile)
    summary={'out_dir':str(OUT),'windows_dir':str(WIN),'device':str(DEVICE),'probe1':rule_summary,'probe2':left_summary,'probe3':profile,'probe4':matrix}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_report(rule_summary,left_summary,profile,matrix)
    copy_outputs()
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

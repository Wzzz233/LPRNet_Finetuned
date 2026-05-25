#!/usr/bin/env python3
"""Generate full StageB1A-E1 moderate LMH extreme dataset.

This is the accepted 20260426 moderate-angle geometry, scaled up for manifest use.
Outputs train/proxy pools with non-overlap controlled later by manifest builder.
"""
import argparse, csv, json, math, random, shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/wzzz/LPRNet')
EXACT_DIR = ROOT / 'datasets/green_exact_quad_synthetic_v1'
LABELS = EXACT_DIR / 'manifests/train_synthetic_labels.txt'
BG_DIR = ROOT / 'tmp/green_extreme_pathB_probe_v2_stronger_20260425/backgrounds'
STATS = ROOT / 'reports/stageB_extreme_detector_vs_gt_stats_20260426/green_det_vs_gt_rows_valid.csv'
OUT = ROOT / 'tmp/green_extreme_stageB1A_E1_moderate_lmh_20260426'
PLATE = (246, 72)
CANVAS = (460, 300)
TIERS = ['low', 'mid', 'high']
DIRECTIONS = ['left_up','left_mid','left_down','right_up','right_mid','right_down','mid_up','mid_down']
PROVINCES = list('京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新')
TIER_PARAMS = {
    'low':  dict(ratio=(2.05,2.45), height=(82,104), yaw_side=(0.80,0.94), pitch=(0.84,1.06), shear=(3,9),   roll=(-4,4),  min_area=11500, min_edge=46, min_angle_gap=4,  max_angle_gap=24),
    'mid':  dict(ratio=(1.92,2.35), height=(82,104), yaw_side=(0.66,0.84), pitch=(0.72,1.12), shear=(9,20),  roll=(-7,7),  min_area=11000, min_edge=42, min_angle_gap=12, max_angle_gap=34),
    'high': dict(ratio=(1.86,2.28), height=(82,104), yaw_side=(0.56,0.74), pitch=(0.64,1.14), shear=(16,28), roll=(-8,8),  min_area=10500, min_edge=42, min_angle_gap=24, max_angle_gap=46),
}


def order_quad(pts):
    pts=np.asarray(pts,dtype=np.float32).reshape(4,2)
    c=pts.mean(axis=0)
    ang=np.arctan2(pts[:,1]-c[1],pts[:,0]-c[0])
    ordered=pts[np.argsort(ang)]
    start=int(np.argmin(ordered.sum(axis=1)))
    ordered=np.roll(ordered,-start,axis=0)
    if ordered[1,0] < ordered[3,0]:
        ordered=np.array([ordered[0],ordered[3],ordered[2],ordered[1]],np.float32)
    return ordered.astype(np.float32)


def parse_quad(name):
    ps=Path(name).stem.split('-')
    if len(ps)>=4 and ps[3].count('&')==4:
        pts=[]
        for pair in ps[3].split('_'):
            x,y=pair.split('&'); pts.append([float(x),float(y)])
        return order_quad(pts)
    return None


def angle(v): return math.degrees(math.atan2(float(v[1]), float(v[0])))


def qstats(q):
    q=order_quad(q)
    top=q[1]-q[0]; right=q[2]-q[1]; bottom=q[2]-q[3]; left=q[3]-q[0]
    edges=[float(np.linalg.norm(top)),float(np.linalg.norm(right)),float(np.linalg.norm(bottom)),float(np.linalg.norm(left))]
    area=float(abs(cv2.contourArea(q.astype(np.float32))))
    ratio=max(edges[0],edges[2])/max(edges[1],edges[3])
    vals=[abs(angle(top)),abs(angle(bottom)),abs(abs(angle(left))-90),abs(abs(angle(right))-90)]
    return {'edges':edges,'area':area,'ratio':ratio,'min_edge':min(edges),'angle_score':max(vals),'angle_mean':sum(vals)/4,'top_abs':vals[0],'bottom_abs':vals[1],'left_dev':vals[2],'right_dev':vals[3]}


def valid_quad(q, min_edge, min_area):
    s=qstats(q)
    if s['min_edge'] < min_edge or s['area'] < min_area:
        return False
    q=order_quad(q)
    return all(np.linalg.norm(q[i]-q[j]) >= min_edge for i in range(4) for j in range(i+1,4))


def load_stats():
    with STATS.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def contains(poly, exact, margin=-1.5):
    poly=order_quad(poly); exact=order_quad(exact)
    return all(cv2.pointPolygonTest(poly,(float(p[0]),float(p[1])),True)>=margin for p in exact)


def expand_cover(pseudo, exact, pad=4, max_iter=4):
    pseudo=order_quad(pseudo).copy(); exact=order_quad(exact)
    for _ in range(max_iter):
        if contains(pseudo, exact): return pseudo, True
        c=pseudo.mean(axis=0)
        for p in exact:
            d=cv2.pointPolygonTest(pseudo,(float(p[0]),float(p[1])),True)
            if d < -1.5:
                idx=int(np.argmin(np.linalg.norm(pseudo-p,axis=1)))
                v=p-c; n=np.linalg.norm(v)+1e-6
                pseudo[idx]=p+(v/n)*pad
        pseudo=order_quad(pseudo)
    return pseudo, contains(pseudo, exact)


def pseudo_loose(q, rng, stats, tier):
    r=rng.choice(stats); s=qstats(q)
    w=max(s['edges'][0],s['edges'][2]); h=max(s['edges'][1],s['edges'][3])
    amp={'low':(0.55,0.90),'mid':(0.70,1.05),'high':(0.78,1.15)}[tier]
    a=rng.uniform(*amp)
    ds=[]
    for name in ['tl','tr','br','bl']:
        dx=np.clip(float(r[f'{name}_dx_n'])*a,-0.18,0.18)
        dy=np.clip(float(r[f'{name}_dy_n'])*a,-0.22,0.22)
        ds.append([dx*w,dy*h])
    pq=order_quad(q)+np.array(ds,np.float32)
    pq[:,0]=np.clip(pq[:,0],0,CANVAS[0]-1); pq[:,1]=np.clip(pq[:,1],0,CANVAS[1]-1)
    pq,ok=expand_cover(pq,q)
    return pq if ok else None


def target_quad(rng,hdir,vdir,tier):
    p=TIER_PARAMS[tier]
    th=rng.uniform(*p['height']); ratio=rng.uniform(*p['ratio']); tw=th*ratio
    side_far=rng.uniform(*p['yaw_side']); side_near=rng.uniform(1.00,1.16)
    if hdir=='left': left_h=th*side_near; right_h=th*side_far
    elif hdir=='right': left_h=th*side_far; right_h=th*side_near
    else: left_h=th*rng.uniform(0.82,1.02); right_h=th*rng.uniform(0.82,1.02)
    pitch_low,pitch_high=p['pitch']
    if vdir=='up': top_w=tw*rng.uniform(pitch_low,0.82); bottom_w=tw*rng.uniform(1.00,pitch_high); shear=-rng.uniform(*p['shear'])
    elif vdir=='down': top_w=tw*rng.uniform(1.00,pitch_high); bottom_w=tw*rng.uniform(pitch_low,0.82); shear=rng.uniform(*p['shear'])
    else: top_w=tw*rng.uniform(0.88,1.06); bottom_w=tw*rng.uniform(0.88,1.06); shear=rng.choice([-1,1])*rng.uniform(p['shear'][0]*0.4,p['shear'][1]*0.7)
    cx=rng.uniform(185,275); cy=rng.uniform(105,195)
    tl=np.array([cx-top_w/2,cy-left_h/2+shear],np.float32)
    tr=np.array([cx+top_w/2,cy-right_h/2-shear],np.float32)
    br=np.array([cx+bottom_w/2,cy+right_h/2-shear*0.35],np.float32)
    bl=np.array([cx-bottom_w/2,cy+left_h/2+shear*0.35],np.float32)
    q=order_quad(np.array([tl,tr,br,bl],np.float32))
    roll=math.radians(rng.uniform(*p['roll']))
    c=q.mean(axis=0); R=np.array([[math.cos(roll),-math.sin(roll)],[math.sin(roll),math.cos(roll)]],np.float32)
    return order_quad((q-c)@R.T+c)


def gate(q,tier):
    p=TIER_PARAMS[tier]; s=qstats(q)
    if not (p['ratio'][0]-0.18 <= s['ratio'] <= p['ratio'][1]+0.22): return False
    if s['area']<p['min_area'] or s['min_edge']<p['min_edge']: return False
    if s['angle_score']<p['min_angle_gap'] or s['angle_score']>p['max_angle_gap']: return False
    if q[:,0].min()<22 or q[:,1].min()<22 or q[:,0].max()>CANVAS[0]-22 or q[:,1].max()>CANVAS[1]-22: return False
    return True


def load_labels():
    rows=[]
    with LABELS.open(encoding='utf-8') as f:
        for line in f:
            ps=line.strip().split()
            if len(ps)>=2 and (EXACT_DIR/ps[0]).exists() and len(ps[1])==8 and ps[1][0] in PROVINCES:
                rows.append((EXACT_DIR/ps[0], ps[1]))
    return rows


def choose_label(labels_by_prov, prov, rng):
    arr=labels_by_prov.get(prov) or []
    if not arr:
        # green_exact_quad_synthetic_v1 has no 安徽 source; for the small original
        # StageB1A 皖 quota, use another province plate texture but stamp a valid
        # generated 安徽 text in the filename/label. Geometry/background is the variable here.
        arr=[x for bucket in labels_by_prov.values() for x in bucket]
    return rng.choice(arr)


def synth_text_for_prov(prov, rng):
    letters='ABCDEFGHJKLMNPQRSTUVWXYZ'
    nums='0123456789'
    # Green8-compatible broad template: province + letter + D/F/A/B/C/E/G/H/J/K + 5 alnum/digits mix.
    third=rng.choice('DFABCEGHJK')
    tail=''.join(rng.choice(nums if i%2 else letters+nums) for i in range(5))
    return prov + rng.choice(letters) + third + tail


def make_one(rng, stats, bgs, labels_by_prov, prov, tier, direction, split, idx):
    hdir,vdir=direction.split('_')
    attempts=0
    while attempts < 5000:
        attempts += 1
        pth,src_text = choose_label(labels_by_prov, prov, rng)
        text = src_text if src_text[0] == prov else synth_text_for_prov(prov, rng)
        img=cv2.imread(str(pth)); srcq=parse_quad(pth.name)
        if img is None or srcq is None: continue
        rect=np.float32([[0,0],[PLATE[0]-1,0],[PLATE[0]-1,PLATE[1]-1],[0,PLATE[1]-1]])
        plate=cv2.warpPerspective(img,cv2.getPerspectiveTransform(srcq,rect),PLATE,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        q=target_quad(rng,hdir,vdir,tier)
        if not gate(q,tier): continue
        bg=cv2.resize(rng.choice(bgs),CANVAS)
        warped=cv2.warpPerspective(plate,cv2.getPerspectiveTransform(rect,q),CANVAS,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
        mask=np.zeros(CANVAS[::-1],np.float32); cv2.fillPoly(mask,[q.astype(np.int32)],1.0); mask=cv2.GaussianBlur(mask,(7,7),2.0)[...,None]
        out=(warped*mask+bg*(1-mask)).astype(np.uint8)
        pq=pseudo_loose(q,rng,stats,tier)
        if pq is None or not valid_quad(pq,12.0,3500.0) or not contains(pq,q): continue
        s=qstats(q); bbox=f'{int(q[:,0].min())}&{int(q[:,1].min())}_{int(q[:,0].max())}&{int(q[:,1].max())}'
        qstr='_'.join(f'{int(round(x))}&{int(round(y))}' for x,y in q)
        fname=f'E1mod-{bbox}-{qstr}-{split}-{tier}-{direction}-{prov}-{idx:04d}-{text}.jpg'
        return fname,out,{'file':fname,'split':split,'tier':tier,'direction':direction,'province':prov,'text':text,'source_exact':str(pth),'exact_quad':q.tolist(),'pseudo_quad':pq.tolist(),**{k:s[k] for k in ['ratio','area','min_edge','angle_score','angle_mean','top_abs','bottom_abs','left_dev','right_dev']},'attempts':attempts}
    raise RuntimeError(f'failed make_one prov={prov} tier={tier} direction={direction} split={split}')


def allocate_train_quota(total_by_prov, tier_targets):
    # deterministic largest-remainder allocation per tier, each province receives sum equal to original quota
    provs=sorted(total_by_prov)
    weights={p:total_by_prov[p] for p in provs}
    total=sum(weights.values())
    tier_alloc={t:{} for t in tier_targets}
    for t,target in tier_targets.items():
        raw=[(p, weights[p]*target/total) for p in provs]
        base={p:int(math.floor(v)) for p,v in raw}
        rem=target-sum(base.values())
        order=sorted(raw, key=lambda x:(x[1]-math.floor(x[1]), x[0]), reverse=True)
        for p,_ in order[:rem]: base[p]+=1
        tier_alloc[t]=base
    # repair per-province sums to original quota by moving between tiers, preserving global tier totals.
    for _ in range(10000):
        diff={p:weights[p]-sum(tier_alloc[t][p] for t in tier_targets) for p in provs}
        if all(v==0 for v in diff.values()): break
        p_plus=next((p for p,v in diff.items() if v>0),None)
        p_minus=next((p for p,v in diff.items() if v<0),None)
        if p_plus is None or p_minus is None: break
        # move one allocation in the same tier from p_minus to p_plus, prefer low/mid over high neutrality
        for t in ['low','mid','high']:
            if tier_alloc[t][p_minus] > 0:
                tier_alloc[t][p_minus]-=1; tier_alloc[t][p_plus]+=1; break
    assert {t:sum(v.values()) for t,v in tier_alloc.items()} == tier_targets
    assert {p:sum(tier_alloc[t][p] for t in tier_targets) for p in provs} == weights
    return tier_alloc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20260426)
    args=ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT/'images').mkdir(parents=True)
    rng=random.Random(args.seed)
    stats=load_stats()
    bgs=[cv2.imread(str(p)) for p in sorted(BG_DIR.glob('bg_*.jpg'))]
    bgs=[b for b in bgs if b is not None]
    labels=load_labels(); labels_by_prov={p:[] for p in PROVINCES}
    for row in labels: labels_by_prov[row[1][0]].append(row)
    if not stats or not bgs or not labels:
        raise SystemExit('missing stats/backgrounds/labels')
    train_quota={'云': 10, '京': 10, '冀': 10, '吉': 10, '宁': 10, '川': 10, '新': 10, '晋': 10, '桂': 10, '沪': 11, '津': 10, '浙': 10, '渝': 10, '湘': 10, '琼': 10, '甘': 10, '皖': 8, '粤': 10, '苏': 10, '蒙': 10, '藏': 10, '豫': 9, '贵': 9, '赣': 9, '辽': 9, '鄂': 9, '闽': 9, '陕': 10, '青': 9, '鲁': 9, '黑': 9}
    train_tier_targets={'low':120,'mid':120,'high':60}
    train_alloc=allocate_train_quota(train_quota, train_tier_targets)
    proxy_quota={p:4 for p in train_quota}
    proxy_tier_targets={'low':48,'mid':48,'high':28}
    proxy_alloc=allocate_train_quota(proxy_quota, proxy_tier_targets)
    records=[]
    for split,alloc in [('train',train_alloc),('proxy',proxy_alloc)]:
        for tier in TIERS:
            per_tier_idx=0
            direction_counts=Counter()
            for prov,cnt in sorted(alloc[tier].items()):
                for _ in range(cnt):
                    direction=DIRECTIONS[per_tier_idx % len(DIRECTIONS)]
                    per_tier_idx += 1
                    direction_counts[direction] += 1
                    fname,img,rec=make_one(rng,stats,bgs,labels_by_prov,prov,tier,direction,split,per_tier_idx)
                    sub=OUT/'images'/split/tier
                    sub.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(sub/fname),img)
                    rec['file']=str((sub/fname).relative_to(OUT))
                    records.append(rec)
            print(split,tier,'count',sum(alloc[tier].values()),'directions',dict(direction_counts))
    meta={'count':len(records),'out':str(OUT),'tier_params':TIER_PARAMS,'train_alloc':train_alloc,'proxy_alloc':proxy_alloc,'records':records,
          'summary':{'by_split':dict(Counter(r['split'] for r in records)),'by_split_tier':dict(Counter(f"{r['split']}:{r['tier']}" for r in records)),'by_split_prov':dict(Counter(f"{r['split']}:{r['province']}" for r in records))}}
    (OUT/'generation_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    keys=['file','split','tier','direction','province','text','source_exact','ratio','area','min_edge','angle_score','angle_mean','top_abs','bottom_abs','left_dev','right_dev','attempts']
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows([{k:r.get(k,'') for k in keys} for r in records])
    print(json.dumps({'generated':len(records),'out':str(OUT),'summary':meta['summary']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()

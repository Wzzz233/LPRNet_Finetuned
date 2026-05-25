#!/usr/bin/env python3
import csv, json, math, random, sys
from pathlib import Path
from collections import Counter
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/wzzz/LPRNet')
sys.path.insert(0,str(ROOT/'src'))
from load_data import UnifiedManifestDataset

MANIFEST_DIR=ROOT/'manifests/curriculum_gray3_stageb_v1_difficulty'
PRED=ROOT/'reports/stageB1A_extreme_QA/extreme_proxy_predictions.csv'
BAD=ROOT/'reports/stageB1A_extreme_QA/extreme_bad_cases_selected.csv'
OUT=ROOT/'reports/stageB1A_extreme_QA_corrected_input'
WIN=Path('/mnt/c/Users/Wzzz2/Desktop/stageB1A_extreme_QA_corrected_input')
OUT.mkdir(parents=True,exist_ok=True); WIN.mkdir(parents=True,exist_ok=True)
FONT_PATH=next(p for p in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc','/usr/share/fonts/opentype/unifont/unifont.otf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists())
FONT=ImageFont.truetype(FONT_PATH,18); TINY=ImageFont.truetype(FONT_PATH,13)

def read_csv(p):
    with open(p,encoding='utf-8') as f: return [dict(r) for r in csv.DictReader(f)]

def denorm_dataset_image(row, split='test'):
    # Build a one-row manifest preserving the original row fields, then call the exact UnifiedManifestDataset path.
    tmp=OUT/'_one_tmp.csv'
    fields=list(row.keys())
    if 'split' not in fields: fields.append('split')
    row2=dict(row); row2['split']=split
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerow({k:row2.get(k,'') for k in fields})
    ds=UnifiedManifestDataset(str(tmp), img_size=[94,24], lpr_max_len=8, split_filter=split,
        ocr_channel_order='bgr', ocr_crop_mode='obb_warp', ocr_resize_mode='letterbox', ocr_resize_kernel='nn',
        ocr_preproc='gray3', ocr_min_occ_ratio=0.90, ocr_quad_pad_ratio=0.0, gray3_prob=1.0)
    arr,label,ln,fam=ds[0]
    img=((np.transpose(arr,(1,2,0))/0.0078125)+127.5).clip(0,255).astype('uint8')
    return img

def pil_bgr(img): return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def crop_thumb(path, size=(235,86)):
    img=cv2.imread(path); h,w=img.shape[:2]
    scale=min(size[0]/w,size[1]/h); nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
    im=pil_bgr(cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA))
    can=Image.new('RGB',size,(245,245,245)); can.paste(im,((size[0]-nw)//2,(size[1]-nh)//2)); return can

def input_thumb(row, size=(235,60), split='test'):
    img=denorm_dataset_image(row, split=split)
    return pil_bgr(cv2.resize(img,size,interpolation=cv2.INTER_NEAREST))

def bbox_nonblack(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); ys,xs=np.where(g>8)
    if len(xs)==0: return None
    return (int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))

def render(items,title,out_path,split='test',cols=3):
    cell_w=330; cell_h=210; title_h=62; rows=math.ceil(len(items)/cols)
    can=Image.new('RGB',(cols*cell_w,title_h+rows*cell_h),(255,255,255)); d=ImageDraw.Draw(can)
    d.text((12,8),title,font=FONT,fill=(0,0,0)); d.text((12,36),f'上: full crop before resize; 下: EXACT UnifiedManifestDataset output 94x24 gray3, enlarged. font={FONT_PATH}',font=TINY,fill=(70,70,70))
    for i,r in enumerate(items):
        x=(i%cols)*cell_w; y=title_h+(i//cols)*cell_h
        d.rectangle([x,y,x+cell_w-1,y+cell_h-1],outline=(200,200,200))
        can.paste(crop_thumb(r['img_path']),(x+6,y+6))
        inp=denorm_dataset_image(r,split=split); can.paste(pil_bgr(cv2.resize(inp,(235,60),interpolation=cv2.INTER_NEAREST)),(x+6,y+98))
        bb=bbox_nonblack(inp)
        d.text((x+6,y+162),f'{i:02d} {Path(r["img_path"]).parent.name} full={cv2.imread(r["img_path"]).shape[1]}x{cv2.imread(r["img_path"]).shape[0]} input_bbox={bb}',font=TINY,fill=(0,0,0))
        if 'gt' in r:
            ok='OK' if r.get('exact')=='True' else 'BAD'; fc='F1OK' if r.get('first_ok')=='True' else 'F1BAD'
            d.text((x+6,y+180),f'GT {r.get("gt")} P {r.get("pred")} {ok}/{fc}',font=TINY,fill=(160,0,0) if ok=='BAD' else (0,120,0))
        else:
            d.text((x+6,y+180),f'TXT {r.get("text")}',font=TINY,fill=(0,0,0))
    can.save(out_path,quality=92)

def strat(rows,n,seed):
    rng=random.Random(seed); by={}
    for r in rows: by.setdefault((r.get('gt') or r.get('text') or '')[0],[]).append(r)
    out=[]; keys=sorted(by)
    while len(out)<n and any(by.values()):
        for k in keys:
            if by[k] and len(out)<n:
                rng.shuffle(by[k]); out.append(by[k].pop())
    return out

pred=read_csv(PRED); bad=read_csv(BAD)
# canonical rows for train/test include exact manifest fields
train=[r for r in read_csv(MANIFEST_DIR/'train_B1A.csv') if r.get('source')=='green_edgefit_extreme']
test=read_csv(MANIFEST_DIR/'proxy_green_edgefit_extreme.csv')
first_bad=[r for r in pred if r.get('first_ok')!='True']
ok=[r for r in pred if r.get('exact')=='True']
sets=[
 ('wrong_cases', bad, 'test', 30),
 ('firstchar_wrong', strat(first_bad,30,2), 'test', 30),
 ('train_extreme', strat(train,30,3), 'train', 30),
 ('test_extreme', strat(test,30,4), 'test', 30),
 ('exact_ok_contrast', ok, 'test', 9),
]
files=[]
for name,rows,split,n in sets:
    rows=rows[:n]
    p=OUT/f'corrected_{name}_fullcrop_and_actual_input_utf8.jpg'
    render(rows, f'B1A extreme {name}: FULL CROP + ACTUAL DATASET 94x24 INPUT', p, split=split, cols=3)
    files.append(str(p))
html='<!doctype html><html><head><meta charset="utf-8"><title>B1A corrected extreme QA</title><style>body{font-family:sans-serif;background:#eee} img{display:block;max-width:100%;margin:18px auto;border:1px solid #999} h1,h2{margin-left:20px}</style></head><body><h1>B1A corrected extreme QA: full crop + actual dataset input</h1>'
for p in files:
    html += f'<h2>{Path(p).name}</h2><img src="{Path(p).name}">\n'
html+='</body></html>'
(OUT/'index.html').write_text(html,encoding='utf-8')
summary={'out_dir':str(OUT),'windows_out_dir':str(WIN),'font_path':FONT_PATH,'files':files,'note':'These sheets use UnifiedManifestDataset._transform_plain exactly for plain_plate rows; no letterbox centering is applied for these synthetic support crops.'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
for p in OUT.glob('*'):
    if p.is_file(): (WIN/p.name).write_bytes(p.read_bytes())
print(json.dumps(summary,ensure_ascii=False,indent=2))

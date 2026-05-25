#!/usr/bin/env python3
import csv, json, math, os, random, sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/training', ROOT/'src/utils', ROOT/'src/evaluation']:
    sys.path.insert(0, str(p))
from load_data import UnifiedManifestDataset, CHARS
from test_LPRNet import collate_fn
from eval_lpr_detailed import decode_logits
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from train_LPRNet import _select_family_logits_from_dict

MODEL = ROOT/'experiments/curriculum_gray3_stageB_v1_B1A_difficulty_conservativeLPRNet__iteration_2000.pth'
MANIFEST_DIR = ROOT/'manifests/curriculum_gray3_stageb_v1_difficulty'
PROXY_EXTREME = MANIFEST_DIR/'proxy_green_edgefit_extreme.csv'
TRAIN_EXTREME = MANIFEST_DIR/'train_B1A.csv'
OUT = ROOT/'reports/stageB1A_extreme_QA'
OUT.mkdir(parents=True, exist_ok=True)
WIN_OUT = Path('/mnt/c/Users/Wzzz2/Desktop/stageB1A_extreme_QA')
WIN_OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDS = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/unifont/unifont.otf',
    '/usr/share/fonts/opentype/unifont/unifont_jp.otf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
font_path = next((p for p in FONT_CANDS if Path(p).exists()), None)
if not font_path:
    raise RuntimeError('No usable font found')
FONT = ImageFont.truetype(font_path, 18)
SMALL = ImageFont.truetype(font_path, 15)
TINY = ImageFont.truetype(font_path, 13)


def read_csv(p):
    with open(p, encoding='utf-8') as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state = torch.load(str(MODEL), map_location=device)
    net, _cfg = build_lprnet_multihead_from_state_dict(state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    load_multihead_state_dict_compat(net, state, strict=False)
    net.to(device).eval()
    return net, device


def predict_proxy():
    ds = UnifiedManifestDataset(
        manifest_path=str(PROXY_EXTREME), img_size=[94,24], lpr_max_len=8, split_filter='test',
        ocr_channel_order='bgr', ocr_crop_mode='obb_warp', ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn', ocr_preproc='gray3', ocr_min_occ_ratio=0.90, ocr_quad_pad_ratio=0.0,
    )
    idx = [i for i,r in enumerate(ds.records) if r.get('family')=='green8' and Path(r.get('img_path','')).exists()]
    loader = DataLoader(Subset(ds, idx), batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)
    net, device = load_model()
    rows=[]
    recs=[ds.records[i] for i in idx]
    cursor=0
    with torch.no_grad():
        for images, labels, lengths, families in loader:
            starts=[]; targets=[]; s=0
            for length in lengths:
                gt=''.join(CHARS[int(c)] for c in labels[s:s+length].numpy().tolist())
                targets.append(gt); s += length
            images=images.to(device)
            raw=net(images)
            logits=_select_family_logits_from_dict(raw, sample_families=list(families)).detach().cpu().numpy()
            dec=decode_logits(logits, 'family_aware_beam', 20, 12, sample_families=list(families))
            for pred_ids, gt in zip(dec, targets):
                pred=''.join(CHARS[int(c)] for c in pred_ids)
                rec=dict(recs[cursor]); cursor += 1
                rec['gt']=gt; rec['pred']=pred
                rec['exact']=str(pred==gt)
                rec['first_ok']=str(bool(pred) and pred[0]==gt[0])
                rec['pos2_ok']=str(len(pred)>1 and pred[1]==gt[1])
                tail_ok=sum(1 for a,b in zip(pred[2:], gt[2:]) if a==b)
                rec['tail_ok']=str(tail_ok)
                rec['tail_total']=str(max(0,len(gt)-2))
                rows.append(rec)
    return rows


def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def read_img(path):
    img=cv2.imread(path)
    if img is None:
        raise RuntimeError(f'cannot read {path}')
    return img


def make_board_view(path):
    # The edgefit rows are plain plate images. For QA, show the model-facing board-like tensor:
    # resize to 94x24 with nn and gray3, then enlarge for human review.
    img=read_img(path)
    resized=cv2.resize(img, (94,24), interpolation=cv2.INTER_NEAREST)
    gray=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray3=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray3


def thumb(path, size=(188,80)):
    img=read_img(path)
    h,w=img.shape[:2]
    scale=min(size[0]/w, size[1]/h)
    nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
    im=bgr_to_pil(cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA))
    canvas=Image.new('RGB', size, (245,245,245))
    canvas.paste(im, ((size[0]-nw)//2,(size[1]-nh)//2))
    return canvas


def board_thumb(path, size=(188,48)):
    img=make_board_view(path)
    im=bgr_to_pil(cv2.resize(img, size, interpolation=cv2.INTER_NEAREST))
    return im


def geom_stats(path):
    img=read_img(path)
    h,w=img.shape[:2]
    # black/near-black area proxy: useful for extreme black-border inspection
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black=float((gray<12).mean())
    board=make_board_view(path)
    bg=float((cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)<12).mean())
    return {'w':w,'h':h,'black_ratio':black,'board_black_ratio':bg}


def render_sheet(items, title, out_path, cols=4):
    cell_w=260; cell_h=190
    title_h=58
    rows=math.ceil(len(items)/cols)
    canvas=Image.new('RGB',(cols*cell_w, title_h+rows*cell_h),(255,255,255))
    d=ImageDraw.Draw(canvas)
    d.text((12,10), title, font=FONT, fill=(0,0,0))
    d.text((12,34), f'n={len(items)} font={font_path}', font=TINY, fill=(80,80,80))
    for i,it in enumerate(items):
        x=(i%cols)*cell_w; y=title_h+(i//cols)*cell_h
        d.rectangle([x,y,x+cell_w-1,y+cell_h-1], outline=(200,200,200))
        p=it['img_path']
        canvas.paste(thumb(p), (x+6,y+6))
        canvas.paste(board_thumb(p), (x+6,y+90))
        txt1=f"{i:02d} {it.get('split','')} {Path(p).parent.name}"
        d.text((x+6,y+140), txt1, font=TINY, fill=(0,0,0))
        if 'gt' in it:
            ok='OK' if it.get('exact')=='True' else 'BAD'
            fc='F1OK' if it.get('first_ok')=='True' else 'F1BAD'
            d.text((x+6,y+155), f"GT {it['gt']}  P {it.get('pred','')} {ok}/{fc}", font=TINY, fill=(160,0,0) if ok=='BAD' else (0,120,0))
        else:
            d.text((x+6,y+155), f"TXT {it.get('text','')}", font=TINY, fill=(0,0,0))
        gs=geom_stats(p)
        d.text((x+6,y+170), f"img {gs['w']}x{gs['h']} black={gs['black_ratio']:.2f} board_black={gs['board_black_ratio']:.2f}", font=TINY, fill=(50,50,50))
    canvas.save(out_path, quality=92)


def stratified_sample(rows, n, seed, require=None):
    if require:
        rows=[r for r in rows if require(r)]
    rng=random.Random(seed)
    by={}
    for r in rows:
        by.setdefault((r.get('text') or r.get('gt') or '')[:1],[]).append(r)
    out=[]
    provs=sorted(by)
    while len(out)<n and any(by.values()):
        for p in provs:
            if by[p] and len(out)<n:
                rng.shuffle(by[p])
                out.append(by[p].pop())
    return out


def main():
    pred_rows=predict_proxy()
    bad=[r for r in pred_rows if r['exact']!='True']
    first_bad=[r for r in pred_rows if r['first_ok']!='True']
    exact_ok=[r for r in pred_rows if r['exact']=='True']
    # Choose enough wrong cases, stratified by province; include rare exact-ok for contrast.
    bad_sample=stratified_sample(bad, 48, 11)
    first_bad_sample=stratified_sample(first_bad, 32, 12)
    ok_sample=stratified_sample(exact_ok, 8, 13)

    train_rows=[r for r in read_csv(TRAIN_EXTREME) if r.get('source')=='green_edgefit_extreme' and r.get('split')=='train']
    test_rows=read_csv(PROXY_EXTREME)
    train_sample=stratified_sample(train_rows, 48, 21)
    test_sample=stratified_sample(test_rows, 48, 22)

    # write CSVs
    fields=sorted(set().union(*(r.keys() for r in pred_rows)))
    with (OUT/'extreme_proxy_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(pred_rows)
    with (OUT/'extreme_bad_cases_selected.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(bad_sample)

    render_sheet(bad_sample, 'B1A extreme proxy wrong cases: original + model-input gray3 94x24', OUT/'contact_sheet_B1A_extreme_wrong_cases_utf8.jpg', cols=4)
    render_sheet(first_bad_sample, 'B1A extreme proxy FIRST-CHAR wrong cases', OUT/'contact_sheet_B1A_extreme_firstchar_wrong_utf8.jpg', cols=4)
    render_sheet(ok_sample, 'B1A extreme proxy exact-correct contrast samples', OUT/'contact_sheet_B1A_extreme_exact_ok_contrast_utf8.jpg', cols=4)
    render_sheet(train_sample, 'B1A TRAIN green_edgefit_extreme samples', OUT/'contact_sheet_B1A_extreme_train_samples_utf8.jpg', cols=4)
    render_sheet(test_sample, 'B1A TEST/PROXY green_edgefit_extreme samples', OUT/'contact_sheet_B1A_extreme_test_samples_utf8.jpg', cols=4)

    # Copy to Windows Desktop QA dir
    for p in OUT.glob('*'):
        if p.is_file():
            data=p.read_bytes()
            (WIN_OUT/p.name).write_bytes(data)

    stats={
        'model': str(MODEL),
        'proxy_manifest': str(PROXY_EXTREME),
        'train_manifest': str(TRAIN_EXTREME),
        'out_dir': str(OUT),
        'windows_out_dir': str(WIN_OUT),
        'font_path': font_path,
        'proxy_total': len(pred_rows),
        'proxy_exact_ok': len(exact_ok),
        'proxy_bad': len(bad),
        'proxy_first_bad': len(first_bad),
        'province_counts_proxy': dict(Counter(r['gt'][0] for r in pred_rows)),
        'province_bad_counts': dict(Counter(r['gt'][0] for r in bad)),
        'generated_files': sorted(str(p) for p in OUT.glob('*')),
    }
    (OUT/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    (WIN_OUT/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

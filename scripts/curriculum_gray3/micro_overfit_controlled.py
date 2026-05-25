#!/usr/bin/env python3
"""Controlled micro overfit: E2 final → fine-tune on 300 E6A only vs 300 E6B only.
Same init, same LR, same epochs. Single variable = data (E6A vs E6B)."""
import csv, json, sys, copy
from pathlib import Path
import numpy as np, torch, cv2
ROOT=Path('/home/wzzz/LPRNet')
for p in [ROOT/'src', ROOT/'src/evaluation', ROOT/'src/training', ROOT/'src/utils']:
    sys.path.insert(0, str(p))
from load_data import CHARS, prepare_board_ocr_input_from_quad_bgr888
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from train_LPRNet import _select_family_logits_from_dict
from eval_lpr_detailed import decode_logits

DEVICE=torch.device('cuda:0')
BLANK=len(CHARS)-1
N_CLASSES=len(CHARS)
INIT=ROOT/'experiments/curriculum_gray3_stageA_v3_realprimary_A1D_green8_template_auxLPRNet__iteration_2000.pth'
OUT=ROOT/'experiments/stageB1A_E6_cluster3geo_probe_20260427/micro_overfit'
OUT.mkdir(exist_ok=True)

E6A_MANIFEST=ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E6A_single_axis_visible_eval_original/train_B1A_E6A_single_axis_visible_eval_original.csv'
E6B_MANIFEST=ROOT/'manifests/curriculum_gray3_stageb_v1_B1A_E6B_compound_visible_eval_original/train_B1A_E6B_compound_visible_eval_original.csv'
E6A_SRC='green_edgefit_extreme_E6A_single_axis_visible_ccpdboard'
E6B_SRC='green_edgefit_extreme_E6B_compound_visible_ccpdboard'

def load_samples(manifest_path, source_filter):
    QKEYS=['quad_1x','quad_1y','quad_2x','quad_2y','quad_3x','quad_3y','quad_4x','quad_4y']
    rows=list(csv.DictReader(manifest_path.open('r',encoding='utf-8-sig')))
    ext=[r for r in rows if r.get('source')==source_filter]
    samples=[]
    for r in ext:
        img=cv2.imread(r['img_path'], cv2.IMREAD_COLOR)
        vals=[float(r[k]) for k in QKEYS]
        q=np.array(vals,dtype=np.float32).reshape(4,2)
        prep,occ,warped,_,_ = prepare_board_ocr_input_from_quad_bgr888(img, q, 94, 24, 'letterbox', 'nn', 'gray3', 'bgr', quad_pad_ratio=0.0)
        x=(prep.astype('float32')-127.5)*0.0078125
        x=torch.from_numpy(np.transpose(x,(2,0,1)))
        gt=r['text']
        target=[CHARS.index(c) for c in gt]
        samples.append((x,target,len(gt)))
    return samples

def eval_set(net, samples):
    net.eval()
    exact=first=short=last=edit_tot=0; n=0
    with torch.no_grad():
        for x,gt_ids,glen in samples:
            xb=x[None,...].to(DEVICE)
            raw=net(xb)
            logits=_select_family_logits_from_dict(raw, sample_families=['green8'])
            if logits is None: continue
            logits_np=logits.detach().cpu().numpy()[0]
            ids=decode_logits(logits_np[None,...], 'family_aware_beam', 20, 12, sample_families=['green8'])[0]
            beam=''.join(CHARS[int(c)] for c in ids)
            gts=''.join(CHARS[c] for c in gt_ids)
            if beam==gts: exact+=1
            if beam and beam[0]==gts[0]: first+=1
            if len(beam)<len(gts): short+=1
            if beam and gts and beam[-1]==gts[-1]: last+=1
            m,nn=len(gts),len(beam); dp=list(range(nn+1))
            for i in range(1,m+1):
                p0=dp[0]; dp[0]=i
                for j in range(1,nn+1):
                    cur=dp[j]; dp[j]=p0 if gts[i-1]==beam[j-1] else 1+min(p0,dp[j],dp[j-1]); p0=cur
            edit_tot+=dp[nn]; n+=1
    net.train()
    if n==0: return {}
    return {'n':n,'exact':exact/n,'first':first/n,'short':short/n,'last':last/n,'edit':edit_tot/n}

def load_model():
    state=torch.load(INIT, map_location=DEVICE)
    net,cfg=build_lprnet_multihead_from_state_dict(state,lpr_max_len=8,phase=False,class_num=N_CLASSES,dropout_rate=0)
    load_multihead_state_dict_compat(net,state,strict=False)
    return net.to(DEVICE), copy.deepcopy(net.state_dict())

def freeze_for_overfit(net):
    for name,param in net.named_parameters():
        if name.startswith(('backbone.18','backbone.19','backbone.20')) or 'adapter' in name or 'head' in name:
            param.requires_grad=True
        else:
            param.requires_grad=False

def overfit(samples, label, epochs=50, B=48, lr=1e-4):
    net, init_state = load_model()
    freeze_for_overfit(net)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
    ctc=torch.nn.CTCLoss(blank=BLANK, reduction='mean', zero_infinity=True)
    results=[]
    print(f'\n--- {label}: {len(samples)} samples, {epochs} epochs ---')
    print('epoch exact first short last edit')
    for epoch in range(1, epochs+1):
        net.train()
        idxs=list(range(len(samples)))
        np.random.shuffle(idxs)
        for start in range(0, len(samples), B):
            batch=idxs[start:start+B]
            xs=torch.stack([samples[i][0] for i in batch]).to(DEVICE)
            inpl=torch.tensor([samples[i][2] for i in batch], dtype=torch.long)
            tgl=torch.tensor([samples[i][2] for i in batch], dtype=torch.long)
            tg=torch.tensor([c for i in batch for c in samples[i][1]], dtype=torch.long)
            raw=net(xs)
            if isinstance(raw, dict):
                if 'multihead_logits' in raw:
                    logits=raw['multihead_logits']['green8']
                elif 'green8' in raw:
                    logits=raw['green8']
                else:
                    logits=list(raw.values())[0]
            else:
                logits=raw
            if epoch==1 and start==0:
                print(f'raw type={type(raw)} raw keys={list(raw.keys()) if isinstance(raw,dict) else "tensor"} logits shape={logits.shape}')
            logits_ctc=logits.permute(2,0,1).log_softmax(dim=2)
            loss=ctc(logits_ctc, tg, inpl, tgl)
            opt.zero_grad(); loss.backward(); opt.step()
            del xs,logits,logits_ctc,loss,tg,inpl,tgl
        if epoch%5==0 or epoch==1:
            r=eval_set(net, samples)
            r['epoch']=epoch
            results.append(r)
            print(f'{epoch:3d} {r["exact"]:.4f} {r["first"]:.4f} {r["short"]:.4f} {r["last"]:.4f} {r["edit"]:.2f}')
    del net; torch.cuda.empty_cache()
    return results

print('loading E6A samples...')
samples_a=load_samples(E6A_MANIFEST, E6A_SRC)
print(f'E6A: {len(samples_a)}')
print('loading E6B samples...')
samples_b=load_samples(E6B_MANIFEST, E6B_SRC)
print(f'E6B: {len(samples_b)}')

res_a=overfit(samples_a, 'E6A', epochs=50)
res_b=overfit(samples_b, 'E6B', epochs=50)

(OUT/'micro_overfit_controlled.json').write_text(json.dumps({'E6A':res_a,'E6B':res_b},ensure_ascii=False,indent=2),encoding='utf-8')
print('\nFinal:')
for name,res in [('E6A',res_a),('E6B',res_b)]:
    r=res[-1]
    print(f'{name} epoch50: exact={r["exact"]:.4f} first={r["first"]:.4f} short={r["short"]:.4f} last={r["last"]:.4f} edit={r["edit"]:.2f}')

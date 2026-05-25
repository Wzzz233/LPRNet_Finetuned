import argparse
import torch
from load_data import CHARS
from LPRNet_multihead import build_lprnet_multihead, FAMILY_HEADS


def build_checkpoint(src_weights: str, dst_weights: str, mode: str):
    single = torch.load(src_weights, map_location='cpu')
    net = build_lprnet_multihead(lpr_max_len=8, phase='train', class_num=len(CHARS), dropout_rate=0.5)
    model = net.state_dict()

    matched = {k: v.clone() for k, v in single.items() if k in model and v.shape == model[k].shape}
    direct = len(matched)
    mapped = []
    container_keys = [k for k in single.keys() if k.startswith('container.')]
    target_families = FAMILY_HEADS if mode == 'all_families' else ('normal7',)
    for family in target_families:
        for old_key in container_keys:
            new_key = old_key.replace('container.', f'containers.{family}.')
            if new_key in model and single[old_key].shape == model[new_key].shape:
                matched[new_key] = single[old_key].clone()
                mapped.append((family, old_key, new_key))
    model.update(matched)
    net.load_state_dict(model)
    torch.save(net.state_dict(), dst_weights)

    backbone_total = sum(1 for k in model if k.startswith('backbone.'))
    backbone_matched = sum(1 for k in matched if k.startswith('backbone.'))
    print({
        'mode': mode,
        'src': src_weights,
        'dst': dst_weights,
        'direct_match_count': direct,
        'mapped_count': len(mapped),
        'mapped_families': list(target_families),
        'backbone_matched': backbone_matched,
        'backbone_total': backbone_total,
    })
    for family in target_families:
        for suffix in ('weight', 'bias'):
            old_key = f'container.0.{suffix}'
            new_key = f'containers.{family}.0.{suffix}'
            if old_key in single and new_key in net.state_dict():
                ok = torch.allclose(single[old_key], net.state_dict()[new_key], atol=1e-6)
                print({'family': family, 'suffix': suffix, 'equal': bool(ok), 'max_diff': float((single[old_key]-net.state_dict()[new_key]).abs().max().item())})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--mode', choices=['all_families', 'normal7_only'], required=True)
    args = ap.parse_args()
    build_checkpoint(args.src, args.dst, args.mode)

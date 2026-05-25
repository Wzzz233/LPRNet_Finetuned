import argparse
import sys
import torch

from load_data import CHARS
from LPRNet import build_lprnet
from LPRNet_multihead import build_lprnet_multihead, FAMILY_HEADS


def map_single_to_multi(pretrained_dict, model_dict):
    matched = {k: v.clone() for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    direct_match_count = len(matched)
    multihead_mapped = {}
    family_map_counts = {family: 0 for family in FAMILY_HEADS}
    container_keys = [k for k in pretrained_dict.keys() if k.startswith('container.')]
    for family in FAMILY_HEADS:
        for old_key in container_keys:
            new_key = old_key.replace('container.', f'containers.{family}.')
            if new_key in model_dict and pretrained_dict[old_key].shape == model_dict[new_key].shape:
                multihead_mapped[new_key] = pretrained_dict[old_key].clone()
                family_map_counts[family] += 1
    matched.update(multihead_mapped)
    return matched, direct_match_count, family_map_counts, multihead_mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()

    single_dict = torch.load(args.weights, map_location='cpu')
    multi_head = build_lprnet_multihead(lpr_max_len=8, phase='train', class_num=len(CHARS), dropout_rate=0.5)
    model_dict = multi_head.state_dict()

    matched, direct_match_count, family_map_counts, multihead_mapped = map_single_to_multi(single_dict, model_dict)
    model_dict.update(matched)
    multi_head.load_state_dict(model_dict)
    loaded_state = multi_head.state_dict()

    backbone_total = sum(1 for k in loaded_state if k.startswith('backbone.'))
    backbone_matched = sum(1 for k in matched if k.startswith('backbone.'))
    backbone_ratio = backbone_matched / backbone_total if backbone_total else 0.0

    checks = []
    for family in FAMILY_HEADS:
        ok = True
        for suffix in ['weight', 'bias']:
            old_key = f'container.0.{suffix}'
            new_key = f'containers.{family}.0.{suffix}'
            if old_key in single_dict and new_key in loaded_state:
                equal = torch.allclose(single_dict[old_key], loaded_state[new_key], atol=1e-6)
                checks.append((family, suffix, equal, float((single_dict[old_key] - loaded_state[new_key]).abs().max().item())))
                ok = ok and equal
            else:
                checks.append((family, suffix, False, float('inf')))
                ok = False
        print(f'[Check] family={family} mapped_ok={ok}')

    print(f'[Verify] direct_match_count={direct_match_count}')
    print(f'[Verify] multihead_mapped={len(multihead_mapped)} family_map_counts={family_map_counts}')
    print(f'[Verify] backbone_matched={backbone_matched}/{backbone_total} ratio={backbone_ratio:.4f}')

    failed = False
    for family, suffix, equal, max_diff in checks:
        status = 'OK' if equal else 'FAIL'
        print(f'[VerifyHead] {family} {suffix} {status} max_diff={max_diff}')
        if not equal:
            failed = True

    if backbone_ratio <= 0.95:
        print('[Verify] FAIL: backbone match ratio <= 0.95')
        failed = True

    if failed:
        print('VERDICT=FAIL')
        sys.exit(1)
    print('VERDICT=PASS')


if __name__ == '__main__':
    main()

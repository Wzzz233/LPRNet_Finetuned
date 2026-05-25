#!/usr/bin/env python3
"""Green8 guard eval: uses training code's test pipeline.
Loads model and evaluates on val_ccpd2020_green.
"""
import sys, torch
from pathlib import Path

ROOT = Path('/home/wzzz/LPRNet')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/training'))

from train_LPRNet import configure_runtime
from load_data import CHARS
from LPRNet_multihead import build_lprnet_multihead_from_state_dict, load_multihead_state_dict_compat
from test_LPRNet import test_greedy

device = torch.device('cuda:0')

# Load model
ckpt_path = ROOT / 'experiments/b_prime_stage1_real_multidomain_20260510/best_LPRNet_model.pth'
state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
net, cfg = build_lprnet_multihead_from_state_dict(
    state, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0
)
load_multihead_state_dict_compat(net, state, strict=False)
net.to(device)
net.eval()

# Build test args
from argparse import Namespace
test_args = Namespace(
    img_size=[94, 24],
    lpr_max_len=8,
    data_mode='manifest',
    train_manifest='', test_manifest=str(ROOT / 'manifests_rebased/curriculum_gray3/val_ccpd2020_green.csv'),
    train_img_dirs='.', test_img_dirs='.',
    train_txt_file='.', test_txt_file='.',
    txt_file='.',
    dataset_root=str(ROOT),
    test_batch_size=120, num_workers=4,
    cuda=True, dropout_rate=0.0,
    ocr_channel_order='bgr', ocr_crop_mode='obb_warp',
    ocr_resize_mode='letterbox', ocr_resize_kernel='nn',
    ocr_preproc='none', ocr_min_occ_ratio=0.9, ocr_quad_pad_ratio=0.0,
    pretrained_model=str(ckpt_path),
    out_json='',
    bad_case_topk=20,
    head_mode='multihead',
    trainable_families='green8',
    enhanced_green_head='expD',
    gray3_prob=0.0,
    brightness_aug_max=0.0,
    plate_box_aug_mode='none', plate_box_aug_prob=0.0,
    plate_box_aug_x=0.06, plate_box_aug_y=0.12, plate_box_aug_min_iou=0.75,
    pos0_head_cols=0,
    pos0_gray_input=False,
)

# Import and run test
sys.path.insert(0, str(ROOT / 'src/evaluation'))
from test_LPRNet import evaluate as run_eval

result = run_eval(test_args)
print(f"\nGreen8 guard results:")
print(f"  exact_plate_acc: {result.get('exact_plate_acc', 'N/A')}")
print(f"  length_correct_acc: {result.get('length_correct_acc', 'N/A')}")
print(f"  position_accuracy: {result.get('position_accuracy', {})}")

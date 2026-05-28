"""
Batch-invariance regression test for LPRNet forward normalization.

Verifies that:
- LPRNet (single head): model(x1) == model([x1, x2, ...])[0]
- LPRNetMultiHead (multihead): same invariant holds for all output logits
"""

import torch
import numpy as np
import sys
import os

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC_DIR)

from LPRNet import build_lprnet
from LPRNet_multihead import build_lprnet_multihead


def _random_tensor(batch, c, h, w):
    """Random tensor in [-1, 1] range, matching typical LPRNet input."""
    return torch.rand(batch, c, h, w) * 2 - 1


def _decode_ctc_greedy(prebs, blank_idx):
    """Decode CTC greedy output for assertion messages."""
    pred, prev = [], -1
    for t in range(prebs.shape[1]):
        idx = int(prebs[:, t].argmax())
        if idx != prev and idx != blank_idx:
            pred.append(idx)
        prev = idx
    return pred


def test_batch_invariance_lprnet_single():
    """Test LPRNet (single head) batch-invariance with random weights."""
    torch.manual_seed(42)
    LPR_MAX_LEN = 8
    CLASS_NUM = 70
    DROPOUT = 0.0

    # Build a single-head LPRNet in eval mode
    model = build_lprnet(
        lpr_max_len=LPR_MAX_LEN,
        phase='test',
        class_num=CLASS_NUM,
        dropout_rate=DROPOUT,
    )
    model.eval()

    # Random input image
    x1 = _random_tensor(1, 3, 24, 94)   # single image
    x2 = _random_tensor(1, 3, 24, 94)
    x3 = _random_tensor(1, 3, 24, 94)
    xb = torch.cat([x1, x2, x3], dim=0)  # batch of 3

    with torch.no_grad():
        out_single = model(x1)  # [1, CLASS_NUM, LPR_MAX_LEN]
        out_batch = model(xb)   # [3, CLASS_NUM, LPR_MAX_LEN]

    # First sample from batch should match single
    diff = (out_single[0] - out_batch[0]).abs()
    max_diff = diff.max().item()
    assert max_diff < 1e-4, (
        f"LPRNet single vs batch[0]: max_diff={max_diff:.2e} (expected < 1e-4)"
    )

    # Also test bigger batch
    x_big = torch.cat([x1] + [_random_tensor(1, 3, 24, 94) for _ in range(255)], dim=0)
    with torch.no_grad():
        out_big = model(x_big)
    big_diff = (out_single[0] - out_big[0]).abs().max().item()
    assert big_diff < 1e-4, (
        f"LPRNet batch=1 vs batch=256: max_diff={big_diff:.2e} (expected < 1e-4)"
    )

    print(f"  [PASS] LPRNet single-head: batch=1 vs batch=3  max_diff={max_diff:.2e}")
    print(f"  [PASS] LPRNet single-head: batch=1 vs batch=256 max_diff={big_diff:.2e}")


def test_batch_invariance_lprnet_multihead():
    """Test LPRNetMultiHead batch-invariance with random weights."""
    torch.manual_seed(42)
    LPR_MAX_LEN = 8
    CLASS_NUM = 70
    DROPOUT = 0.0

    # Build a basic multihead model (no pos0, no province, no adapters)
    model = build_lprnet_multihead(
        lpr_max_len=LPR_MAX_LEN,
        phase='test',
        class_num=CLASS_NUM,
        dropout_rate=DROPOUT,
        enhanced_green_head=False,
        pos0_head_cols=0,
        adapter_families=None,
        enable_shared_province_head=False,
    )
    model.eval()

    x1 = _random_tensor(1, 3, 24, 94)
    x2 = _random_tensor(1, 3, 24, 94)
    xb = torch.cat([x1, x2], dim=0)
    x_big = torch.cat([x1] + [_random_tensor(1, 3, 24, 94) for _ in range(255)], dim=0)

    with torch.no_grad():
        out_single = model(x1)    # dict with family keys
        out_batch = model(xb)     # same keys
        out_big = model(x_big)

    # Test batch-invariance for each family output
    all_pass = True
    for key in out_single:
        diff3 = (out_single[key][0] - out_batch[key][0]).abs().max().item()
        diff256 = (out_single[key][0] - out_big[key][0]).abs().max().item()
        ok = diff3 < 1e-4 and diff256 < 1e-4
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] MultiHead {key}: batch=1vs2={diff3:.2e}  1vs256={diff256:.2e}")
        if not ok:
            all_pass = False
    assert all_pass, "MultiHead baseline batch-invariance failed"


def test_batch_invariance_multihead_with_aux():
    """Test LPRNetMultiHead with pos0 head and family adapters — batch-invariant."""
    torch.manual_seed(42)
    LPR_MAX_LEN = 8
    CLASS_NUM = 70
    DROPOUT = 0.0

    # Multihead with pos0_head and family adapters
    model = build_lprnet_multihead(
        lpr_max_len=LPR_MAX_LEN,
        phase='test',
        class_num=CLASS_NUM,
        dropout_rate=DROPOUT,
        enhanced_green_head=False,
        pos0_head_cols=4,
        pos0_num_classes=31,
        adapter_families=['green8'],
        adapter_hidden_channels=128,
        enable_shared_province_head=False,
    )
    model.eval()

    x1 = _random_tensor(1, 3, 24, 94)
    xb = torch.cat([x1, _random_tensor(1, 3, 24, 94)], dim=0)
    x_big = torch.cat([x1] + [_random_tensor(1, 3, 24, 94) for _ in range(255)], dim=0)

    with torch.no_grad():
        out1 = model(x1)
        outb = model(xb)
        out_big = model(x_big)

    all_pass = True
    for key in out1:
        diff2 = (out1[key][0] - outb[key][0]).abs().max().item()
        diff256 = (out1[key][0] - out_big[key][0]).abs().max().item()
        ok = diff2 < 1e-4 and diff256 < 1e-4
        status = "PASS" if ok else "FAIL"
        shape_str = 'x'.join(str(s) for s in out1[key].shape)
        print(f"  [{status}] MultiHead+aux {key} ({shape_str}): batch=1vs2={diff2:.2e}  1vs256={diff256:.2e}")
        if not ok:
            all_pass = False
    assert all_pass, "MultiHead+aux batch-invariance failed"


if __name__ == '__main__':
    print("=== LPRNet Batch-Invariance Regression Tests ===")
    print()

    test_batch_invariance_lprnet_single()
    test_batch_invariance_lprnet_multihead()
    test_batch_invariance_multihead_with_aux()

    print()
    print("All tests passed.")

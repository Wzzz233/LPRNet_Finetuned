#!/usr/bin/env python3
"""
Build yellow model (class_num=70) and migrate weights from blue expert (class_num=68)
by character name, not by positional index.
"""
import sys, os
sys.path.insert(0, '/home/wzzz/LPRNet/src')
sys.path.insert(0, '/home/wzzz/LPRNet/src/training')
import torch
from LPRNet import build_lprnet

# CHARS definitions
BLUE_CHARS = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
              '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
              '桂','琼','川','贵','云','藏','陕','甘','青','宁','新',
              '0','1','2','3','4','5','6','7','8','9',
              'A','B','C','D','E','F','G','H','J','K',
              'L','M','N','P','Q','R','S','T','U','V',
              'W','X','Y','Z','I','O','-']  # 68 chars, blank='-' at index 67

YELLOW_CHARS = ['京','沪','津','渝','冀','晋','蒙','辽','吉','黑',
                '苏','浙','皖','闽','赣','鲁','豫','鄂','湘','粤',
                '桂','琼','川','贵','云','藏','陕','甘','青','宁','新',
                '0','1','2','3','4','5','6','7','8','9',
                'A','B','C','D','E','F','G','H','J','K',
                'L','M','N','P','Q','R','S','T','U','V',
                'W','X','Y','Z','I','O',
                '学','挂','-']  # 70 chars, blank='-' at index 69

# Map character → index for both charsets
blue_map = {c: i for i, c in enumerate(BLUE_CHARS)}
yellow_map = {c: i for i, c in enumerate(YELLOW_CHARS)}

def build_yellow_model():
    """Create yellow LPRNet with class_num=70."""
    net = build_lprnet(lpr_max_len=8, phase=False, class_num=70, dropout_rate=0)
    return net

def migrate_from_blue(yellow_net, blue_path):
    """
    Load blue expert weights, migrate by character name to yellow model.
    
    Layers to migrate per-character:
    - backbone.20.weight / .bias:   Conv2d(256, class_num, 13, 1)
    - backbone.21.weight / .bias / .running_mean / .running_var: BN(class_num)
    - container.0.weight / .bias: Conv2d(448+class_num, class_num, 1, 1)
    
    Other layers: direct copy (same shapes)
    """
    blue_state = torch.load(blue_path, map_location='cpu')
    yellow_state = yellow_net.state_dict()
    
    # Build index map: for each YELLOW char, find its index in BLUE
    # fill_value = blue index if exists, else None (random init)
    char_to_blue = {}
    for yc in YELLOW_CHARS:
        char_to_blue[yc] = blue_map.get(yc)  # None if not in blue (学, 挂)
    
    # layer name → (output_dim_is_char, input_dim_is_char_for_container)
    # Layers with class_num-dependent shapes:
    char_layers = ['backbone.20.weight', 'backbone.20.bias',
                   'backbone.21.weight', 'backbone.21.bias',
                   'backbone.21.running_mean', 'backbone.21.running_var',
                   'container.0.weight', 'container.0.bias']
    
    migrated_count = 0
    random_init_count = 0
    
    for yellow_key in yellow_state:
        if yellow_key in char_layers:
            # Per-character migration needed
            blue_key = yellow_key
            blue_t = blue_state[blue_key]
            yellow_t = yellow_state[yellow_key]
            yellow_cls = 70
            blue_cls = 68
            
            if yellow_key.endswith('.weight') and 'container' in yellow_key:
                # container.0.weight: shape (out_cls, 448+out_cls, 1, 1)
                # Need to migrate BOTH output AND input dims
                out_dim = yellow_cls
                in_dim = 448 + yellow_cls
                
                # Start with zeros
                new_w = torch.zeros_like(yellow_t)
                
                # Copy input channels 0-447 (shared features) for each output char
                for y_idx in range(out_dim):
                    y_char = YELLOW_CHARS[y_idx]
                    b_idx = char_to_blue[y_char]
                    if b_idx is not None:
                        # Output row: copy from blue
                        # Input cols 0-447: copy from blue (shared features)
                        new_w[y_idx, :448, :, :] = blue_t[b_idx, :448, :, :]
                        # Input cols 448+: copy per-character
                        for b_ic in range(blue_cls):
                            b_ic_char = BLUE_CHARS[b_ic]
                            # Find where this char in yellow
                            if b_ic_char in yellow_map:
                                y_ic = yellow_map[b_ic_char]
                                new_w[y_idx, 448 + y_ic, :, :] = blue_t[b_idx, 448 + b_ic, :, :]
                        migrated_count += 1
                    else:
                        random_init_count += 1
                        # Leave as random (from xavier init in yellow model)
                
                yellow_state[yellow_key].copy_(new_w)
                
            elif yellow_key.endswith('.weight') and 'backbone.20' in yellow_key:
                # backbone.20.weight: (out_cls, 256, 13, 1)
                new_w = torch.zeros_like(yellow_t)
                for y_idx in range(yellow_cls):
                    y_char = YELLOW_CHARS[y_idx]
                    b_idx = char_to_blue[y_char]
                    if b_idx is not None:
                        new_w[y_idx] = blue_t[b_idx]
                        migrated_count += 1
                    else:
                        random_init_count += 1
                yellow_state[yellow_key].copy_(new_w)
                
            elif yellow_key.endswith('.bias') or 'running_mean' in yellow_key or 'running_var' in yellow_key:
                # 1D tensors: (class_num,)
                new_t = torch.zeros_like(yellow_t)
                for y_idx in range(yellow_cls):
                    y_char = YELLOW_CHARS[y_idx]
                    b_idx = char_to_blue[y_char]
                    if b_idx is not None:
                        new_t[y_idx] = blue_t[b_idx]
                        migrated_count += 1
                    else:
                        random_init_count += 1
                yellow_state[yellow_key].copy_(new_t)
        else:
            # Direct copy (same shape)
            if yellow_key in blue_state and yellow_state[yellow_key].shape == blue_state[yellow_key].shape:
                yellow_state[yellow_key].copy_(blue_state[yellow_key])
                migrated_count += 1
            # Skip keys not in blue (num_batches_tracked) or shape mismatch
    
    yellow_net.load_state_dict(yellow_state)
    print(f'Weight migration: {migrated_count} layers/rows copied, {random_init_count} randomly initialized')
    
    # Verify: count how many char rows are inherited vs random
    with torch.no_grad():
        w20 = yellow_net.state_dict()['backbone.20.weight']
        inherited = sum(1 for yc in YELLOW_CHARS if char_to_blue[yc] is not None)
        random_chars = [yc for yc in YELLOW_CHARS if char_to_blue[yc] is None]
        print(f'Chars inherited from blue: {inherited}/70 ({", ".join(random_chars)} randomly init)')
    
    return yellow_net

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/home/wzzz/LPRNet/experiments/special_yellow_v5/init_from_blue.pth')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    net = build_yellow_model()
    net = migrate_from_blue(net, '/home/wzzz/LPRNet/models/weights/weights_official/Final_LPRNet_model.pth')
    
    torch.save(net.state_dict(), args.output)
    print(f'Saved initialized model to {args.output}')

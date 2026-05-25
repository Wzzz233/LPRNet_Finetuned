import torch


def is_cjk(tok):
    if not tok:
        return False
    cp = ord(tok[0])
    return 0x4E00 <= cp <= 0x9FFF


def fuse_first_char(base_text, aux_char, aux_conf, mode, threshold):
    if not base_text or not aux_char:
        return base_text, False, 'empty'
    if mode == 'replace_all':
        if base_text[0] == aux_char:
            return base_text, False, 'same'
        return aux_char + base_text[1:], True, 'replace_all'
    if mode == 'replace_if_not_cjk':
        if is_cjk(base_text[0]):
            return base_text, False, 'base_is_cjk'
        if base_text[0] == aux_char:
            return base_text, False, 'same'
        return aux_char + base_text[1:], True, 'replace_if_not_cjk'
    if mode == 'replace_if_confident':
        if aux_conf < threshold:
            return base_text, False, 'below_threshold'
        if base_text[0] == aux_char:
            return base_text, False, 'same'
        return aux_char + base_text[1:], True, 'replace_if_confident'
    return base_text, False, 'fusion_disabled'


def extract_branch_logits(raw_dict, families, shared_key, family_prefix):
    if shared_key in raw_dict and raw_dict[shared_key] is not None:
        return raw_dict[shared_key]
    selected = []
    template = None
    for key, branch in raw_dict.items():
        if key.startswith(f'{family_prefix}_') and branch is not None:
            template = branch[:1]
            break
    if template is None:
        return None
    for i, family in enumerate(families):
        key = f'{family_prefix}_{family}'
        branch = raw_dict.get(key)
        if branch is None:
            selected.append(torch.zeros_like(template))
            continue
        selected.append(branch[i:i + 1])
    return torch.cat(selected, dim=0) if selected else None


def extract_pos0_logits(raw_dict, families):
    return extract_branch_logits(raw_dict, families, shared_key='pos0', family_prefix='pos0')


def extract_province_logits(raw_dict, families):
    return extract_branch_logits(raw_dict, families, shared_key='province', family_prefix='province')

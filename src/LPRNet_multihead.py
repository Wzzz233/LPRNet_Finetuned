import torch
import torch.nn as nn

FAMILY_HEADS = ('normal7', 'green8', 'special')

DEFAULT_POS0_NUM_CLASSES = 31  # 省份字数量；后续含使/领/W时可扩展到34+
DEFAULT_PROVINCE_NUM_CLASSES = 31
DEFAULT_CLASS_NUM = 66


def _extract_families_from_prefix(state_dict, prefix):
    families = set()
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        family = suffix.split('.', 1)[0].strip()
        if family in FAMILY_HEADS:
            families.add(family)
    return sorted(families)


def infer_checkpoint_class_num(state_dict, default_class_num=DEFAULT_CLASS_NUM):
    primary_candidates = set()
    aux_candidates = set()

    for key in ('backbone.20.weight', 'backbone.21.weight', 'backbone.21.bias', 'backbone.21.running_mean', 'backbone.21.running_var'):
        tensor = state_dict.get(key)
        if tensor is not None and getattr(tensor, 'ndim', 0) >= 1 and int(tensor.shape[0]) > 0:
            primary_candidates.add(int(tensor.shape[0]))

    for key, tensor in state_dict.items():
        if getattr(tensor, 'ndim', 0) != 4:
            continue
        if key.startswith('containers.') and key.endswith('.0.weight'):
            in_ch = int(tensor.shape[1])
            if in_ch >= 448:
                primary_candidates.add(in_ch - 448)
            continue
        if key in ('pos0_head.0.weight', 'province_head.0.weight'):
            in_ch = int(tensor.shape[1])
            if in_ch >= 448:
                aux_candidates.add(in_ch - 448)
            continue
        if any(key.startswith(prefix) and key.endswith('.0.weight') for prefix in ('family_adapters.', 'pos0_family_heads.', 'province_family_heads.')):
            in_ch = int(tensor.shape[1])
            if in_ch >= 448:
                aux_candidates.add(in_ch - 448)

    if len(primary_candidates) > 1:
        raise RuntimeError(f'inconsistent primary checkpoint class_num candidates: {sorted(primary_candidates)}')
    if primary_candidates:
        return int(next(iter(primary_candidates)))
    if len(aux_candidates) > 1:
        raise RuntimeError(f'inconsistent aux checkpoint class_num candidates: {sorted(aux_candidates)}')
    if aux_candidates:
        return int(next(iter(aux_candidates)))
    return int(default_class_num)


def infer_checkpoint_multihead_config(state_dict, default_pos0_head_cols=4, default_class_num=DEFAULT_CLASS_NUM):
    adapter_families = _extract_families_from_prefix(state_dict, 'family_adapters.')
    pos0_target_families = _extract_families_from_prefix(state_dict, 'pos0_family_heads.')
    province_target_families = _extract_families_from_prefix(state_dict, 'province_family_heads.')
    slot_target_families = _extract_families_from_prefix(state_dict, 'slot_family_heads.')
    has_shared_pos0 = 'pos0_head.4.weight' in state_dict
    has_shared_province = 'province_head.4.weight' in state_dict
    pos0_num_classes = DEFAULT_POS0_NUM_CLASSES
    province_num_classes = DEFAULT_PROVINCE_NUM_CLASSES
    if has_shared_pos0:
        pos0_num_classes = int(state_dict['pos0_head.4.weight'].shape[0])
    elif pos0_target_families:
        class_counts = {
            int(state_dict[f'pos0_family_heads.{family}.4.weight'].shape[0])
            for family in pos0_target_families
            if f'pos0_family_heads.{family}.4.weight' in state_dict
        }
        if len(class_counts) > 1:
            raise RuntimeError(f'inconsistent family-specific pos0 class counts: {sorted(class_counts)}')
        if class_counts:
            pos0_num_classes = int(next(iter(class_counts)))

    if has_shared_province:
        province_num_classes = int(state_dict['province_head.4.weight'].shape[0])
    elif province_target_families:
        class_counts = {
            int(state_dict[f'province_family_heads.{family}.4.weight'].shape[0])
            for family in province_target_families
            if f'province_family_heads.{family}.4.weight' in state_dict
        }
        if len(class_counts) > 1:
            raise RuntimeError(f'inconsistent family-specific province class counts: {sorted(class_counts)}')
        if class_counts:
            province_num_classes = int(next(iter(class_counts)))

    enhanced_green_head = ''
    class_num = infer_checkpoint_class_num(state_dict, default_class_num=default_class_num)
    green8_conv = state_dict.get('containers.green8.0.weight')
    if green8_conv is not None and getattr(green8_conv, 'ndim', 0) == 4 and tuple(green8_conv.shape[2:]) == (3, 3):
        out_channels = int(green8_conv.shape[0])
        if out_channels == 256:
            enhanced_green_head = 'expD'
        elif out_channels == 512:
            enhanced_green_head = 'expE'

    has_any_pos0 = bool(has_shared_pos0 or pos0_target_families)
    has_any_province = bool(has_shared_province or province_target_families)
    return {
        'enhanced_green_head': enhanced_green_head,
        'adapter_families': adapter_families,
        'pos0_target_families': pos0_target_families,
        'province_target_families': province_target_families,
        'slot_target_families': slot_target_families,
        'has_shared_pos0': has_shared_pos0,
        'has_any_pos0': has_any_pos0,
        'has_shared_province': has_shared_province,
        'has_any_province': has_any_province,
        'class_num': int(class_num),
        'pos0_head_cols': int(default_pos0_head_cols) if has_any_pos0 else 0,
        'pos0_num_classes': int(pos0_num_classes),
        'province_num_classes': int(province_num_classes),
    }


def adapt_legacy_aux_state_dict(state_dict, model_state_dict):
    aux_prefixes = ('family_adapters.', 'pos0_head.', 'province_head.', 'pos0_family_heads.', 'province_family_heads.', 'slot_family_heads.')
    patched = {}
    adapted_keys = []
    for key, value in state_dict.items():
        target = model_state_dict.get(key)
        if target is None or tuple(value.shape) == tuple(target.shape):
            patched[key] = value
            continue
        if not key.startswith(aux_prefixes) or getattr(value, 'ndim', None) != getattr(target, 'ndim', None):
            patched[key] = value
            continue
        resized = target.detach().clone()
        resized.zero_()
        common_slices = tuple(slice(0, min(int(src_dim), int(dst_dim))) for src_dim, dst_dim in zip(value.shape, target.shape))
        resized[common_slices] = value[common_slices].to(dtype=target.dtype)
        patched[key] = resized
        adapted_keys.append((key, tuple(value.shape), tuple(target.shape)))
    return patched, adapted_keys


def load_multihead_state_dict_compat(net, state_dict, strict=False):
    patched_state, adapted_keys = adapt_legacy_aux_state_dict(state_dict, net.state_dict())
    return net.load_state_dict(patched_state, strict=strict), adapted_keys


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class SharedBackboneLPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super().__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
        )

    def extract_context(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)
        global_context = []
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)
        return torch.cat(global_context, 1)


class LPRNetMultiHead(SharedBackboneLPRNet):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate, enhanced_green_head=False,
                 pos0_head_cols=0, pos0_num_classes=DEFAULT_POS0_NUM_CLASSES, adapter_families=None,
                 adapter_hidden_channels=128, enable_shared_province_head=False,
                 province_num_classes=DEFAULT_PROVINCE_NUM_CLASSES):
        super().__init__(lpr_max_len, phase, class_num, dropout_rate)
        self.enhanced_green_head = enhanced_green_head
        self.containers = nn.ModuleDict()
        self.pos0_head_cols = pos0_head_cols
        self.adapter_families = set(adapter_families or [])
        self.family_adapters = nn.ModuleDict()
        for family in FAMILY_HEADS:
            if family == 'green8' and enhanced_green_head == 'expE':
                # 实验E：三层更深 head (516→512→256→68)
                self.containers[family] = nn.Sequential(
                    nn.Conv2d(in_channels=448 + self.class_num, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Conv2d(in_channels=256, out_channels=self.class_num, kernel_size=(1, 1)),
                )
            elif family == 'green8' and enhanced_green_head:
                # 实验D：两层 head (516→256→68)
                self.containers[family] = nn.Sequential(
                    nn.Conv2d(in_channels=448 + self.class_num, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(in_channels=256, out_channels=self.class_num, kernel_size=(1, 1)),
                )
            else:
                # 标准单层 head
                self.containers[family] = nn.Sequential(
                    nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
                )

        in_ch = 448 + self.class_num
        for family in self.adapter_families:
            self.family_adapters[family] = nn.Sequential(
                nn.Conv2d(in_ch, adapter_hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(adapter_hidden_channels, in_ch, kernel_size=1),
                nn.ReLU(inplace=True),
            )

        # 全 family 共享的 pos0 分类头（省份字 / 特殊 pos0 字符）
        if pos0_head_cols > 0:
            self.pos0_head = nn.Sequential(
                nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, pos0_num_classes),
            )
        else:
            self.pos0_head = None
        self.pos0_family_heads = nn.ModuleDict()

        if enable_shared_province_head:
            self.province_head = nn.Sequential(
                nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, province_num_classes),
            )
        else:
            self.province_head = None
        self.province_family_heads = nn.ModuleDict()
        self.slot_family_heads = nn.ModuleDict()

    def enable_family_specific_pos0(self, families, pos0_num_classes=DEFAULT_POS0_NUM_CLASSES):
        in_ch = 448 + self.class_num
        self.pos0_family_heads = nn.ModuleDict()
        for family in families:
            self.pos0_family_heads[family] = nn.Sequential(
                nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, pos0_num_classes),
            )

    def enable_family_specific_province(self, families, province_num_classes=DEFAULT_PROVINCE_NUM_CLASSES):
        in_ch = 448 + self.class_num
        self.province_family_heads = nn.ModuleDict()
        for family in families:
            self.province_family_heads[family] = nn.Sequential(
                nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, province_num_classes),
            )

    def enable_family_specific_slot(self, families):
        in_ch = 448 + self.class_num
        self.slot_family_heads = nn.ModuleDict()
        for family in families:
            self.slot_family_heads[family] = nn.Sequential(
                nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, self.lpr_max_len)),
                nn.Conv2d(128, self.class_num, kernel_size=1),
            )

    def forward(self, x, pos0_input=None):
        context = self.extract_context(x)
        out = {}
        family_contexts = {}
        for family, container in self.containers.items():
            family_context = self.family_adapters[family](context) if family in self.family_adapters else context
            family_contexts[family] = family_context
            head = container(family_context)
            out[family] = torch.mean(head, dim=2)
        if len(self.pos0_family_heads) > 0:
            pos0_context = self.extract_context(pos0_input) if pos0_input is not None else context
            for family, head in self.pos0_family_heads.items():
                family_pos0_context = self.family_adapters[family](pos0_context) if family in self.family_adapters else pos0_context
                out[f'pos0_{family}'] = head(family_pos0_context[:, :, :, :self.pos0_head_cols])
        elif self.pos0_head is not None:
            pos0_context = self.extract_context(pos0_input) if pos0_input is not None else context
            out['pos0'] = self.pos0_head(pos0_context[:, :, :, :self.pos0_head_cols])

        if len(self.province_family_heads) > 0:
            for family, head in self.province_family_heads.items():
                family_province_context = family_contexts.get(family, context)
                out[f'province_{family}'] = head(family_province_context)
        elif self.province_head is not None:
            out['province'] = self.province_head(context)
        if len(self.slot_family_heads) > 0:
            for family, head in self.slot_family_heads.items():
                family_slot_context = family_contexts.get(family, context)
                slot_logits = head(family_slot_context).squeeze(2).permute(0, 2, 1).contiguous()
                out[f'slot_{family}'] = slot_logits
        return out


def build_lprnet_multihead(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5, enhanced_green_head=False,
                           pos0_head_cols=0, pos0_num_classes=DEFAULT_POS0_NUM_CLASSES, adapter_families=None,
                           adapter_hidden_channels=128, enable_shared_province_head=False,
                           province_num_classes=DEFAULT_PROVINCE_NUM_CLASSES):
    net = LPRNetMultiHead(lpr_max_len, phase, class_num, dropout_rate, enhanced_green_head=enhanced_green_head,
                          pos0_head_cols=pos0_head_cols, pos0_num_classes=pos0_num_classes,
                          adapter_families=adapter_families, adapter_hidden_channels=adapter_hidden_channels,
                          enable_shared_province_head=enable_shared_province_head,
                          province_num_classes=province_num_classes)
    if phase is True or phase == 'train':
        return net.train()
    return net.eval()


def build_lprnet_multihead_from_state_dict(state_dict, lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5,
                                           default_pos0_head_cols=4, adapter_hidden_channels=128):
    cfg = infer_checkpoint_multihead_config(
        state_dict,
        default_pos0_head_cols=default_pos0_head_cols,
        default_class_num=class_num,
    )
    net = build_lprnet_multihead(
        lpr_max_len=lpr_max_len,
        phase=phase,
        class_num=cfg['class_num'],
        dropout_rate=dropout_rate,
        enhanced_green_head=cfg['enhanced_green_head'],
        pos0_head_cols=cfg['pos0_head_cols'],
        pos0_num_classes=cfg['pos0_num_classes'],
        adapter_families=cfg['adapter_families'],
        adapter_hidden_channels=adapter_hidden_channels,
        enable_shared_province_head=cfg['has_shared_province'],
        province_num_classes=cfg['province_num_classes'],
    )
    if cfg['pos0_target_families']:
        net.enable_family_specific_pos0(cfg['pos0_target_families'], pos0_num_classes=cfg['pos0_num_classes'])
    if cfg['province_target_families']:
        net.enable_family_specific_province(cfg['province_target_families'], province_num_classes=cfg['province_num_classes'])
    if cfg['slot_target_families']:
        net.enable_family_specific_slot(cfg['slot_target_families'])
    return net, cfg

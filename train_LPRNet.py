# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, PROVINCE_COUNT, LPRDataLoader, CCPDBoardDataLoader, BoardDumpDataLoader
from model.LPRNet import build_lprnet
from test_LPRNet import greedy_decode_logits
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = base_lr * (0.1 ** len(lr_schedule))
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=15, type=int, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], nargs=2, type=int, help='the image size')
    parser.add_argument('--train_img_dirs', default="./balanced_ccpd_red_ppm", help='the train images path')
    parser.add_argument('--test_img_dirs', default="./balanced_ccpd_red_ppm", help='the test images path')
    parser.add_argument('--txt_file', default="./balanced_ccpd_red_ppm/train_labels.txt", help='legacy shared label txt file path')
    parser.add_argument('--train_txt_file', default=None, help='train label txt file path')
    parser.add_argument('--test_txt_file', default=None, help='test label txt file path')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, type=int, help='license plate number max length.')
    parser.add_argument('--data_mode', default='standard', choices=['standard', 'ccpd_board'], help='dataset preprocessing mode')
    parser.add_argument('--ocr_channel_order', default='bgr', choices=['rgb', 'bgr'], help='board-aligned OCR input order')
    parser.add_argument('--ocr_crop_mode', default='match', choices=['fixed', 'box', 'tight', 'box-pad', 'match'], help='board-aligned OCR crop mode')
    parser.add_argument('--ocr_resize_mode', default='letterbox', choices=['stretch', 'letterbox'], help='board-aligned OCR resize mode')
    parser.add_argument('--ocr_resize_kernel', default='nn', choices=['nn', 'bilinear'], help='board-aligned OCR resize kernel')
    parser.add_argument('--ocr_preproc', default='none', choices=['none', 'raw', 'gray', 'gray3', 'bin'], help='board-aligned OCR crop preprocess')
    parser.add_argument('--ocr_min_occ_ratio', default=0.90, type=float, help='board-aligned recrop threshold')
    parser.add_argument('--train_plate_box_aug_mode', default='none', choices=['none', 'jitter_refine'], help='train-only plate box augmentation profile')
    parser.add_argument('--train_plate_box_aug_prob', default=0.0, type=float, help='probability of applying train-only plate box augmentation')
    parser.add_argument('--train_plate_box_aug_x', default=0.06, type=float, help='train-only horizontal bbox jitter fraction')
    parser.add_argument('--train_plate_box_aug_y', default=0.12, type=float, help='train-only vertical bbox jitter fraction')
    parser.add_argument('--train_plate_box_aug_min_iou', default=0.75, type=float, help='minimum IoU to keep train-only augmented bbox')
    parser.add_argument('--board_anchor_img_dirs', default='.', help='directories for raw board OCR dump images')
    parser.add_argument('--board_anchor_txt_file', default='', help='label txt for raw board OCR dump anchors')
    parser.add_argument('--board_anchor_sample_weight', default=512.0, type=float, help='sampler weight multiplier for board anchor samples')
    parser.add_argument('--pseudo_anchor_img_dirs', default='', help='directories for CCPD pseudo-anchor images')
    parser.add_argument('--pseudo_anchor_train_txt_file', default='', help='label txt for CCPD pseudo-anchor train split')
    parser.add_argument('--pseudo_anchor_val_txt_file', default='', help='label txt for CCPD pseudo-anchor validation split')
    parser.add_argument('--pseudo_anchor_sample_weight', default=192.0, type=float, help='sampler weight multiplier for CCPD pseudo-anchor train samples')
    parser.add_argument('--province_balance_mode', default='inv_sqrt', choices=['none', 'inv_sqrt'], help='province rebalance mode for training sampler and first-char loss')
    parser.add_argument('--first_char_aux_weight', default=0.4, type=float, help='auxiliary loss weight for first province character')
    parser.add_argument('--first_char_time_steps', default=6, type=int, help='number of early time steps used for first-char auxiliary head proxy')
    parser.add_argument('--selection_proxy_eval_samples', default=5000, type=int, help='number of validation samples used to break checkpoint-selection ties once board-anchor metrics are equal')
    parser.add_argument('--train_batch_size', default=64, type=int, help='training batch size.')
    parser.add_argument('--test_batch_size', default=120, type=int, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=str2bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], nargs='+', type=int, help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights_local/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='./weights_red_stage3/Final_LPRNet_model.pth', help='pretrained base model')

    args = parser.parse_args()
    if args.train_txt_file is None:
        args.train_txt_file = args.txt_file
    if args.test_txt_file is None:
        args.test_txt_file = args.train_txt_file

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(int)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


def extract_first_char_targets(labels, lengths):
    first_targets = []
    start = 0
    for length in lengths:
        if length <= 0:
            first_targets.append(len(CHARS) - 1)
        else:
            first_targets.append(int(labels[start]))
        start += length
    return first_targets


def build_province_weights(texts, mode):
    weights = np.ones(PROVINCE_COUNT, dtype=np.float32)
    if mode == 'none':
        return weights
    counts = np.ones(PROVINCE_COUNT, dtype=np.float64)
    for text in texts:
        if not text:
            continue
        idx = CHARS_DICT.get(text[0], -1)
        if 0 <= idx < PROVINCE_COUNT:
            counts[idx] += 1.0
    raw = 1.0 / np.sqrt(counts)
    raw *= (PROVINCE_COUNT / raw.sum())
    return raw.astype(np.float32)


def build_sample_weights(texts, sample_sources, province_mode, board_anchor_sample_weight, pseudo_anchor_sample_weight):
    province_weights = build_province_weights(texts, province_mode)
    sample_weights = []
    for text, source in zip(texts, sample_sources):
        weight = 1.0
        if text:
            idx = CHARS_DICT.get(text[0], -1)
            if 0 <= idx < PROVINCE_COUNT:
                weight *= float(province_weights[idx])
        if source == 'board':
            weight *= float(board_anchor_sample_weight)
        elif source == 'pseudo':
            weight *= float(pseudo_anchor_sample_weight)
        sample_weights.append(weight)
    return np.asarray(sample_weights, dtype=np.float64), province_weights


def make_train_loader(dataset, sample_weights, batch_size, num_workers):
    if sample_weights is None:
        return DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(dataset, batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)


def evaluate_first_char_dataset(net, dataset, batch_size, num_workers, use_cuda, first_char_time_steps, detail_limit=0):
    if dataset is None or len(dataset) == 0:
        return None
    loader = DataLoader(dataset, min(batch_size, max(1, len(dataset))), shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    net.eval()
    exact = 0
    first_ok = 0
    blank_ratio_sum = 0.0
    total = 0
    details = []
    seen = 0
    with torch.no_grad():
        for images, labels, lengths in loader:
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start + length]
                targets.append(label.clone())
                start += length
            if use_cuda:
                images = images.cuda()
            logits = net(images)
            prebs = logits.detach().cpu().numpy()
            decoded = greedy_decode_logits(prebs)
            first_proxy = torch.softmax(logits[:, :PROVINCE_COUNT, :first_char_time_steps].mean(dim=2), dim=1).cpu()
            blank_idx = len(CHARS) - 1
            argmax_t = np.argmax(prebs, axis=1)
            blank_ratio = np.mean(argmax_t == blank_idx, axis=1)

            for i, seq in enumerate(decoded):
                pred_text = ''.join(CHARS[int(c)] for c in seq)
                gt_text = ''.join(CHARS[int(c)] for c in targets[i].tolist())
                exact += int(pred_text == gt_text)
                first_ok += int(bool(gt_text) and bool(pred_text) and gt_text[0] == pred_text[0])
                blank_ratio_sum += float(blank_ratio[i])
                topk = torch.topk(first_proxy[i], k=min(5, PROVINCE_COUNT))
                top5 = [(CHARS[int(idx)], float(val)) for val, idx in zip(topk.values.tolist(), topk.indices.tolist())]
                if detail_limit <= 0 or len(details) < detail_limit:
                    row = {
                        'gt': gt_text,
                        'pred': pred_text,
                        'first_char_top5': top5,
                        'blank_top1_ratio': float(blank_ratio[i]),
                    }
                    img_paths = getattr(dataset, 'img_paths', None)
                    if img_paths is not None and (seen + i) < len(img_paths):
                        row['image_path'] = img_paths[seen + i]
                    details.append(row)
                total += 1
            seen += len(decoded)
    net.train()
    return {
        'sample_count': total,
        'exact_plate_acc': (exact / total) if total else 0.0,
        'first_char_acc': (first_ok / total) if total else 0.0,
        'blank_top1_mean': (blank_ratio_sum / total) if total else 0.0,
        'details': details,
    }


def evaluate_board_anchor_dataset(net, dataset, batch_size, num_workers, use_cuda, first_char_time_steps):
    return evaluate_first_char_dataset(net, dataset, batch_size, num_workers, use_cuda, first_char_time_steps, detail_limit=8)


def better_board_metric(current, best):
    if best is None:
        return True
    current_key = (
        current['first_char_acc'],
        current['exact_plate_acc'],
        current.get('pseudo_first_char_acc', -1.0),
        current.get('pseudo_exact_plate_acc', -1.0),
        current.get('proxy_exact_plate_acc', -1.0),
        -current['blank_top1_mean'],
    )
    best_key = (
        best['first_char_acc'],
        best['exact_plate_acc'],
        best.get('pseudo_first_char_acc', -1.0),
        best.get('pseudo_exact_plate_acc', -1.0),
        best.get('proxy_exact_plate_acc', -1.0),
        -best['blank_top1_mean'],
    )
    return current_key > best_key


def evaluate_exact_plate_subset(net, dataset, batch_size, num_workers, use_cuda, max_samples):
    if dataset is None or len(dataset) == 0 or max_samples <= 0:
        return None
    eval_dataset = dataset
    if max_samples < len(dataset):
        eval_dataset = Subset(dataset, list(range(max_samples)))
    loader = DataLoader(
        eval_dataset,
        batch_size=min(batch_size, len(eval_dataset)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    net.eval()
    exact = 0
    total = 0
    with torch.no_grad():
        for images, labels, lengths in loader:
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length
            if use_cuda:
                images = images.cuda()
            logits = net(images)
            decoded = greedy_decode_logits(logits.detach().cpu().numpy())
            for pred_ids, gt_ids in zip(decoded, targets):
                exact += int(np.array_equal(np.asarray(pred_ids), np.asarray(gt_ids)))
                total += 1
    net.train()
    return (exact / total) if total else 0.0

def train():
    args = get_parser()

    T_length = 18 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")
    print(f"[Env] device={device} model_training={lprnet.training}")

    # load pretrained model (智能微调版)
    if args.pretrained_model:
        pretrained_dict = torch.load(args.pretrained_model, map_location=torch.device('cpu'))
        model_dict = lprnet.state_dict()
        
        # 神奇过滤器：只保留我们模型里有的，并且形状长得一模一样的权重参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # 把过滤好的旧知识更新到现在的模型里
        model_dict.update(pretrained_dict)
        lprnet.load_state_dict(model_dict)
        print(f"【微调模式】成功加载预训练权重！保留了 {len(pretrained_dict)} 个匹配的层。")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    optimizer = optim.RMSprop(
        lprnet.parameters(),
        lr=args.learning_rate,
        alpha=0.9,
        eps=1e-08,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    board_anchor_img_dirs = os.path.expanduser(args.board_anchor_img_dirs)
    pseudo_anchor_img_dirs = os.path.expanduser(args.pseudo_anchor_img_dirs) if args.pseudo_anchor_img_dirs else train_img_dirs

    board_anchor_eval_dataset = None
    pseudo_anchor_val_dataset = None
    train_texts = []
    sample_sources = []

    if args.data_mode == 'ccpd_board':
        common_dataset_kwargs = dict(
            ocr_channel_order=args.ocr_channel_order,
            ocr_crop_mode=args.ocr_crop_mode,
            ocr_resize_mode=args.ocr_resize_mode,
            ocr_resize_kernel=args.ocr_resize_kernel,
            ocr_preproc=args.ocr_preproc,
            ocr_min_occ_ratio=args.ocr_min_occ_ratio,
        )
        train_main_dataset = CCPDBoardDataLoader(
            train_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.train_txt_file,
            plate_box_aug_mode=args.train_plate_box_aug_mode,
            plate_box_aug_prob=args.train_plate_box_aug_prob,
            plate_box_aug_x=args.train_plate_box_aug_x,
            plate_box_aug_y=args.train_plate_box_aug_y,
            plate_box_aug_min_iou=args.train_plate_box_aug_min_iou,
            **common_dataset_kwargs,
        )
        test_dataset = CCPDBoardDataLoader(
            test_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.test_txt_file,
            **common_dataset_kwargs,
        )
    else:
        train_main_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len, txt_file=args.train_txt_file)
        test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len, txt_file=args.test_txt_file)

    if len(train_main_dataset) == 0:
        raise RuntimeError(f"No training samples found. train_img_dirs={args.train_img_dirs} train_txt_file={args.train_txt_file}")
    if len(test_dataset) == 0:
        raise RuntimeError(f"No test samples found. test_img_dirs={args.test_img_dirs} test_txt_file={args.test_txt_file}")

    train_dataset = train_main_dataset
    train_texts.extend(list(getattr(train_main_dataset, 'img_labels', [])))
    sample_sources.extend(['main'] * len(getattr(train_main_dataset, 'img_labels', [])))

    if args.pseudo_anchor_train_txt_file:
        if args.data_mode != 'ccpd_board':
            raise RuntimeError('pseudo anchors require --data_mode ccpd_board')
        pseudo_anchor_train_dataset = CCPDBoardDataLoader(
            pseudo_anchor_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.pseudo_anchor_train_txt_file,
            plate_box_aug_mode=args.train_plate_box_aug_mode,
            plate_box_aug_prob=args.train_plate_box_aug_prob,
            plate_box_aug_x=args.train_plate_box_aug_x,
            plate_box_aug_y=args.train_plate_box_aug_y,
            plate_box_aug_min_iou=args.train_plate_box_aug_min_iou,
            **common_dataset_kwargs,
        )
        if len(pseudo_anchor_train_dataset) > 0:
            train_dataset = ConcatDataset([train_dataset, pseudo_anchor_train_dataset])
            train_texts.extend(list(pseudo_anchor_train_dataset.img_labels))
            sample_sources.extend(['pseudo'] * len(pseudo_anchor_train_dataset))
        else:
            print(f"[PseudoAnchorTrain] no valid samples from {args.pseudo_anchor_train_txt_file}, disable pseudo-anchor training")

    if args.pseudo_anchor_val_txt_file:
        if args.data_mode != 'ccpd_board':
            raise RuntimeError('pseudo anchors require --data_mode ccpd_board')
        pseudo_anchor_val_dataset = CCPDBoardDataLoader(
            pseudo_anchor_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.pseudo_anchor_val_txt_file,
            **common_dataset_kwargs,
        )
        if len(pseudo_anchor_val_dataset) == 0:
            print(f"[PseudoAnchorVal] no valid samples from {args.pseudo_anchor_val_txt_file}, disable pseudo-anchor validation")
            pseudo_anchor_val_dataset = None

    if args.board_anchor_txt_file:
        board_anchor_eval_dataset = BoardDumpDataLoader(
            board_anchor_img_dirs.split(','),
            args.img_size,
            args.lpr_max_len,
            txt_file=args.board_anchor_txt_file,
        )
        if len(board_anchor_eval_dataset) > 0:
            train_dataset = ConcatDataset([train_dataset, board_anchor_eval_dataset])
            train_texts.extend(list(board_anchor_eval_dataset.img_labels))
            sample_sources.extend(['board'] * len(board_anchor_eval_dataset))
        else:
            print(f"[BoardAnchor] no valid samples from {args.board_anchor_txt_file}, disable anchor training")
            board_anchor_eval_dataset = None

    sample_weights, province_weights = build_sample_weights(
        train_texts,
        sample_sources,
        args.province_balance_mode,
        args.board_anchor_sample_weight,
        args.pseudo_anchor_sample_weight,
    )
    province_ce_weights = torch.tensor(province_weights, dtype=torch.float32, device=device)
    train_loader = make_train_loader(train_dataset, sample_weights, args.train_batch_size, args.num_workers)
    epoch_size = len(train_loader)
    if epoch_size == 0:
        raise RuntimeError(
            f"train_batch_size={args.train_batch_size} is larger than effective train samples={len(train_dataset)}; "
            "reduce batch size or add more samples."
        )

    print(
        f"[Data] mode={args.data_mode} train_samples={len(train_dataset)} test_samples={len(test_dataset)} "
        f"train_plate_box_aug={args.train_plate_box_aug_mode} prob={args.train_plate_box_aug_prob:.2f} "
        f"board_anchors={(len(board_anchor_eval_dataset) if board_anchor_eval_dataset else 0)} "
        f"pseudo_train={(sample_sources.count('pseudo'))} pseudo_val={(len(pseudo_anchor_val_dataset) if pseudo_anchor_val_dataset else 0)} "
        f"province_balance={args.province_balance_mode} first_char_aux={args.first_char_aux_weight:.2f}"
    )

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')
    global_iter = args.resume_epoch * epoch_size
    best_board_metric = None
    best_state_dict = None
    best_epoch = 0
    best_proxy_acc = -1.0

    for epoch in range(args.resume_epoch + 1, args.max_epoch + 1):
        lprnet.train()
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)
        train_loader = make_train_loader(train_dataset, sample_weights, args.train_batch_size, args.num_workers)
        epoch_loss_sum = 0.0
        epoch_aux_sum = 0.0
        epoch_loss_count = 0

        for epochiter, (images, labels, lengths) in enumerate(train_loader):
            start_time = time.time()
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
            if args.cuda:
                images = Variable(images, requires_grad=False).cuda()
                labels = Variable(labels, requires_grad=False).cuda()
            else:
                images = Variable(images, requires_grad=False)
                labels = Variable(labels, requires_grad=False)

            logits = lprnet(images)
            log_probs = logits.permute(2, 0, 1).log_softmax(2).requires_grad_()
            aux_steps = max(1, min(args.first_char_time_steps, logits.shape[2]))
            first_targets = torch.tensor(extract_first_char_targets(labels, lengths), dtype=torch.long, device=labels.device)
            first_logits = logits[:, :PROVINCE_COUNT, :aux_steps].mean(dim=2)
            aux_loss = F.cross_entropy(first_logits, first_targets, weight=province_ce_weights)

            optimizer.zero_grad()
            ctc = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            loss = ctc + args.first_char_aux_weight * aux_loss
            loss_item = loss.item()
            aux_item = aux_loss.item()
            if np.isnan(loss_item):
                print("[Fatal] Loss became NaN. Stop training.")
                return
            if loss_item == np.inf:
                continue
            loss.backward()
            optimizer.step()

            global_iter += 1
            epoch_loss_sum += loss_item
            epoch_aux_sum += aux_item
            epoch_loss_count += 1
            end_time = time.time()

            if global_iter % args.save_interval == 0:
                torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(global_iter) + '.pth')

            if global_iter == 1 or epochiter % 20 == 0:
                print(
                    'Epoch:' + repr(epoch) + ' || epochiter: ' + repr(epochiter) + '/' + repr(epoch_size)
                    + '|| Totel iter ' + repr(global_iter) + ' || Loss: %.4f|| Aux: %.4f||' % (loss_item, aux_item)
                    + 'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr)
                )

        avg_epoch_loss = epoch_loss_sum / max(1, epoch_loss_count)
        avg_aux_loss = epoch_aux_sum / max(1, epoch_loss_count)
        print('[Epoch Summary] Epoch {} AvgLoss {:.4f} AvgAux {:.4f}'.format(epoch, avg_epoch_loss, avg_aux_loss))

        board_metrics = evaluate_board_anchor_dataset(
            lprnet,
            board_anchor_eval_dataset,
            args.test_batch_size,
            args.num_workers,
            args.cuda,
            args.first_char_time_steps,
        )
        pseudo_metrics = evaluate_first_char_dataset(
            lprnet,
            pseudo_anchor_val_dataset,
            args.test_batch_size,
            args.num_workers,
            args.cuda,
            args.first_char_time_steps,
            detail_limit=0,
        )
        proxy_exact = evaluate_exact_plate_subset(
            lprnet,
            test_dataset,
            args.test_batch_size,
            args.num_workers,
            args.cuda,
            args.selection_proxy_eval_samples,
        )
        if proxy_exact is not None:
            print(
                '[SelectionProxy] Epoch {} exact {:.4f} on first {} eval samples'.format(
                    epoch,
                    proxy_exact,
                    min(args.selection_proxy_eval_samples, len(test_dataset)),
                )
            )
        if pseudo_metrics is not None:
            print(
                '[PseudoAnchorVal] Epoch {} exact {:.4f} first_char {:.4f} blank_mean {:.4f}'.format(
                    epoch,
                    pseudo_metrics['exact_plate_acc'],
                    pseudo_metrics['first_char_acc'],
                    pseudo_metrics['blank_top1_mean'],
                )
            )
        if board_metrics is not None:
            board_metrics['proxy_exact_plate_acc'] = proxy_exact if proxy_exact is not None else -1.0
            board_metrics['pseudo_first_char_acc'] = pseudo_metrics['first_char_acc'] if pseudo_metrics is not None else -1.0
            board_metrics['pseudo_exact_plate_acc'] = pseudo_metrics['exact_plate_acc'] if pseudo_metrics is not None else -1.0
            print(
                '[BoardAnchor] Epoch {} exact {:.4f} first_char {:.4f} blank_mean {:.4f} pseudo_first {:.4f} pseudo_exact {:.4f} proxy_exact {:.4f}'.format(
                    epoch,
                    board_metrics['exact_plate_acc'],
                    board_metrics['first_char_acc'],
                    board_metrics['blank_top1_mean'],
                    board_metrics['pseudo_first_char_acc'],
                    board_metrics['pseudo_exact_plate_acc'],
                    board_metrics['proxy_exact_plate_acc'],
                )
            )
            for idx, item in enumerate(board_metrics['details'], start=1):
                print(
                    '[BoardAnchor][{}] gt={} pred={} blank={:.4f} top5={}'.format(
                        idx,
                        item['gt'],
                        item['pred'],
                        item['blank_top1_ratio'],
                        item['first_char_top5'],
                    )
                )
            if better_board_metric(board_metrics, best_board_metric):
                best_board_metric = board_metrics
                best_state_dict = {k: v.detach().cpu().clone() for k, v in lprnet.state_dict().items()}
                best_epoch = epoch
                print('[BoardAnchor] New best checkpoint selected at epoch {}'.format(epoch))

    if best_state_dict is not None:
        lprnet.load_state_dict(best_state_dict)
        print('[Training Done] Load best board-anchor checkpoint from epoch {}'.format(best_epoch))

    print("Final test Accuracy:")
    final_acc = Greedy_Decode_Eval(lprnet, test_dataset, args)
    best_proxy_acc = max(best_proxy_acc, final_acc)
    print('[Training Done] Best Test Accuracy: {:.6f}'.format(best_proxy_acc))
    if best_board_metric is not None:
        print(
            '[Training Done] Best board anchor metrics: exact {:.4f} first_char {:.4f} blank_mean {:.4f} pseudo_first {:.4f} pseudo_exact {:.4f} proxy_exact {:.4f}'.format(
                best_board_metric['exact_plate_acc'],
                best_board_metric['first_char_acc'],
                best_board_metric['blank_top1_mean'],
                best_board_metric.get('pseudo_first_char_acc', -1.0),
                best_board_metric.get('pseudo_exact_plate_acc', -1.0),
                best_board_metric.get('proxy_exact_plate_acc', -1.0),
            )
        )

    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    return Acc
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))


if __name__ == "__main__":
    train()

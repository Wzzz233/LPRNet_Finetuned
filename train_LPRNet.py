# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader, CCPDBoardDataLoader
from model.LPRNet import build_lprnet
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

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
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
    if args.data_mode == 'ccpd_board':
        common_dataset_kwargs = dict(
            ocr_channel_order=args.ocr_channel_order,
            ocr_crop_mode=args.ocr_crop_mode,
            ocr_resize_mode=args.ocr_resize_mode,
            ocr_resize_kernel=args.ocr_resize_kernel,
            ocr_preproc=args.ocr_preproc,
            ocr_min_occ_ratio=args.ocr_min_occ_ratio,
        )
        train_dataset = CCPDBoardDataLoader(
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
        train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len, txt_file=args.train_txt_file)
        test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len, txt_file=args.test_txt_file)
    if len(train_dataset) == 0:
        raise RuntimeError(f"No training samples found. train_img_dirs={args.train_img_dirs} train_txt_file={args.train_txt_file}")
    if len(test_dataset) == 0:
        raise RuntimeError(f"No test samples found. test_img_dirs={args.test_img_dirs} test_txt_file={args.test_txt_file}")
    print(
        f"[Data] mode={args.data_mode} train_samples={len(train_dataset)} test_samples={len(test_dataset)} "
        f"train_plate_box_aug={args.train_plate_box_aug_mode} prob={args.train_plate_box_aug_prob:.2f}"
    )

    epoch_size = len(train_dataset) // args.train_batch_size
    if epoch_size == 0:
        raise RuntimeError(
            f"train_batch_size={args.train_batch_size} is larger than train_samples={len(train_dataset)}; "
            "reduce batch size or add more samples."
        )
    max_iter = args.max_epoch * epoch_size

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    epoch_loss_sum = 0.0
    epoch_loss_count = 0
    eval_records = []
    best_acc = -1.0
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            epoch += 1
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth')

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # print(labels)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if args.cuda:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.detach().requires_grad_()
        # print(log_probs.shape)
        # backprop
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        loss_item = loss.item()
        if np.isnan(loss_item):
            print("[Fatal] Loss became NaN. Stop training.")
            return
        if loss_item == np.inf:
            continue
        loss.backward()
        optimizer.step()
        loss_val += loss_item
        epoch_loss_sum += loss_item
        epoch_loss_count += 1
        end_time = time.time()
        if iteration % 20 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss_item) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))

        # epoch-end summary and periodic evaluation (every 10 epochs)
        if (iteration + 1) % epoch_size == 0:
            avg_epoch_loss = epoch_loss_sum / max(1, epoch_loss_count)
            print('[Epoch Summary] Epoch {} AvgLoss {:.4f}'.format(epoch, avg_epoch_loss))
            if epoch % 10 == 0:
                print('[Epoch Eval] Epoch {} start evaluation...'.format(epoch))
                acc = Greedy_Decode_Eval(lprnet, test_dataset, args)
                eval_records.append((epoch, avg_epoch_loss, acc))
                if acc > best_acc:
                    best_acc = acc
                print('[Epoch Eval] Epoch {} AvgLoss {:.4f} TestAcc {:.6f}'.format(epoch, avg_epoch_loss, acc))
    # final test
    print("Final test Accuracy:")
    final_acc = Greedy_Decode_Eval(lprnet, test_dataset, args)
    if final_acc > best_acc:
        best_acc = final_acc
    print('[Training Done] Best Test Accuracy: {:.6f}'.format(best_acc))
    if eval_records:
        print('[Training Done] 10-epoch checkpoints:')
        for ep, avg_loss, acc in eval_records:
            print('  - Epoch {} | AvgLoss {:.4f} | TestAcc {:.6f}'.format(ep, avg_loss, acc))

    # save final parameters
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

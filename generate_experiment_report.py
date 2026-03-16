#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path


def read_json(path_str: str):
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def parse_train_log(path_str: str):
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8', errors='ignore')
    epoch_rows = []
    for epoch, loss, aux in re.findall(r"\[Epoch Summary\] Epoch (\d+) AvgLoss ([0-9.]+)(?: AvgAux ([0-9.]+))?", text):
        epoch_rows.append({
            'epoch': int(epoch),
            'avg_loss': float(loss),
            'avg_aux': float(aux) if aux else None,
        })
    best_proxy = None
    m = re.search(r"\[Training Done\] Best Test Accuracy: ([0-9.]+)", text)
    if m:
        best_proxy = float(m.group(1))
    return {
        'epoch_rows': epoch_rows,
        'best_proxy_acc': best_proxy,
    }


def fmt_pct(x):
    if x is None:
        return 'N/A'
    return f'{x * 100:.2f}%'


def summarize_generalization(val_metrics, test_metrics):
    if not val_metrics or not test_metrics:
        return "验证或测试指标缺失，无法判断泛化差距。"
    gap_exact = val_metrics['exact_plate_acc'] - test_metrics['exact_plate_acc']
    gap_first = val_metrics['province_first_char_acc'] - test_metrics['province_first_char_acc']
    if gap_exact > 0.30 or gap_first > 0.20:
        return (
            f"验证到测试存在明显泛化差距：整牌差距 {gap_exact:.4f}，首字差距 {gap_first:.4f}。"
            " 这更像部署域差或训练分布偏置，而不只是普通过拟合。"
        )
    if gap_exact > 0.10 or gap_first > 0.10:
        return (
            f"验证到测试有中等泛化差距：整牌差距 {gap_exact:.4f}，首字差距 {gap_first:.4f}。"
        )
    return (
        f"验证与测试较为一致：整牌差距 {gap_exact:.4f}，首字差距 {gap_first:.4f}。"
    )


def parse_args():
    ap = argparse.ArgumentParser(description='Generate markdown report for an experiment run.')
    ap.add_argument('--experiment_name', required=True)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--train_txt', required=True)
    ap.add_argument('--val_txt', required=True)
    ap.add_argument('--test_txt', required=True)
    ap.add_argument('--board_anchor_txt', default='')
    ap.add_argument('--data_mode', default='')
    ap.add_argument('--ocr_channel_order', default='')
    ap.add_argument('--ocr_crop_mode', default='')
    ap.add_argument('--ocr_resize_mode', default='')
    ap.add_argument('--ocr_resize_kernel', default='')
    ap.add_argument('--ocr_preproc', default='')
    ap.add_argument('--ocr_min_occ_ratio', default='')
    ap.add_argument('--pretrained_model', default='')
    ap.add_argument('--learning_rate', default='')
    ap.add_argument('--lr_schedule', default='')
    ap.add_argument('--max_epoch', default='')
    ap.add_argument('--train_batch_size', default='')
    ap.add_argument('--test_batch_size', default='')
    ap.add_argument('--train_plate_box_aug_mode', default='')
    ap.add_argument('--train_plate_box_aug_prob', default='')
    ap.add_argument('--train_plate_box_aug_x', default='')
    ap.add_argument('--train_plate_box_aug_y', default='')
    ap.add_argument('--train_plate_box_aug_min_iou', default='')
    ap.add_argument('--province_balance_mode', default='')
    ap.add_argument('--board_anchor_sample_weight', default='')
    ap.add_argument('--first_char_aux_weight', default='')
    ap.add_argument('--first_char_time_steps', default='')
    ap.add_argument('--selection_proxy_eval_samples', default='')
    ap.add_argument('--report_path', default='')
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    report_path = Path(args.report_path) if args.report_path else run_dir / 'EXPERIMENT_REPORT.md'

    split_stats = read_json(str(run_dir / 'split_stats.json'))
    val_metrics = read_json(str(run_dir / 'val_metrics.json'))
    test_metrics = read_json(str(run_dir / 'test_metrics.json'))
    board_metrics = read_json(str(run_dir / 'board_anchor_metrics.json'))
    train_info = parse_train_log(str(run_dir / 'train.log'))

    lines = []
    lines.append(f"# {args.experiment_name}")
    lines.append("")
    lines.append("## 摘要")
    lines.append(f"- 实验目录：`{run_dir}`")
    if val_metrics:
        lines.append(f"- 验证集整牌准确率：`{val_metrics['exact_plate_acc']:.6f}`")
        lines.append(f"- 验证集首字准确率：`{val_metrics['province_first_char_acc']:.6f}`")
    if test_metrics:
        lines.append(f"- 测试集整牌准确率：`{test_metrics['exact_plate_acc']:.6f}`")
        lines.append(f"- 测试集首字准确率：`{test_metrics['province_first_char_acc']:.6f}`")
    if board_metrics:
        lines.append(f"- 真实板图锚点整牌准确率：`{board_metrics['exact_plate_acc']:.6f}`")
        lines.append(f"- 真实板图锚点首字准确率：`{board_metrics['first_char_acc']:.6f}`")
    lines.append("")

    lines.append("## 数据")
    lines.append(f"- 训练标签：`{args.train_txt}`")
    lines.append(f"- 验证标签：`{args.val_txt}`")
    lines.append(f"- 测试标签：`{args.test_txt}`")
    if args.board_anchor_txt:
        lines.append(f"- 真实板图锚点：`{args.board_anchor_txt}`")
    if split_stats:
        lines.append(f"- 训练样本数：`{split_stats['train']['sample_count']}`")
        lines.append(f"- 验证样本数：`{split_stats['val']['sample_count']}`")
        lines.append(f"- 测试样本数：`{split_stats['test']['sample_count']}`")
        lines.append(
            f"- 切分重叠：train/val=`{split_stats['overlap']['train_val']}` "
            f"train/test=`{split_stats['overlap']['train_test']}` val/test=`{split_stats['overlap']['val_test']}`"
        )
    lines.append("")

    lines.append("## 处理与训练设置")
    lines.append(f"- 数据模式：`{args.data_mode}`")
    lines.append(f"- OCR 输入：`channel={args.ocr_channel_order}` `crop={args.ocr_crop_mode}` `resize={args.ocr_resize_mode}` `kernel={args.ocr_resize_kernel}` `preproc={args.ocr_preproc}` `min_occ={args.ocr_min_occ_ratio}`")
    lines.append(f"- 初始化权重：`{args.pretrained_model}`")
    lines.append(f"- 学习率：`{args.learning_rate}`，调度：`{args.lr_schedule}`，epoch：`{args.max_epoch}`")
    lines.append(f"- batch：train=`{args.train_batch_size}` test=`{args.test_batch_size}`")
    if args.train_plate_box_aug_mode:
        lines.append(
            f"- 裁切扰动：`mode={args.train_plate_box_aug_mode}` `prob={args.train_plate_box_aug_prob}` "
            f"`jx={args.train_plate_box_aug_x}` `jy={args.train_plate_box_aug_y}` `min_iou={args.train_plate_box_aug_min_iou}`"
        )
    if args.province_balance_mode:
        lines.append(f"- 省份重平衡：`{args.province_balance_mode}`")
    if args.board_anchor_sample_weight:
        lines.append(f"- 真实锚点采样权重：`{args.board_anchor_sample_weight}`")
    if args.first_char_aux_weight:
        lines.append(f"- 首字辅助损失：`weight={args.first_char_aux_weight}` `time_steps={args.first_char_time_steps}`")
    if args.selection_proxy_eval_samples:
        lines.append(f"- checkpoint 选择代理集：`first {args.selection_proxy_eval_samples} val samples`")
    if train_info.get('epoch_rows'):
        first_loss = train_info['epoch_rows'][0]['avg_loss']
        last_loss = train_info['epoch_rows'][-1]['avg_loss']
        lines.append(f"- 训练损失：首 epoch `AvgLoss={first_loss:.4f}`，末 epoch `AvgLoss={last_loss:.4f}`")
    lines.append("")

    lines.append("## 表现")
    if val_metrics:
        lines.append(
            f"- 验证集：整牌 `{fmt_pct(val_metrics['exact_plate_acc'])}`，首字 `{fmt_pct(val_metrics['province_first_char_acc'])}`，字符 `{fmt_pct(val_metrics['char_acc'])}`"
        )
    if test_metrics:
        lines.append(
            f"- 测试集：整牌 `{fmt_pct(test_metrics['exact_plate_acc'])}`，首字 `{fmt_pct(test_metrics['province_first_char_acc'])}`，字符 `{fmt_pct(test_metrics['char_acc'])}`"
        )
    if board_metrics:
        lines.append(
            f"- 真实板图锚点：整牌 `{fmt_pct(board_metrics['exact_plate_acc'])}`，首字 `{fmt_pct(board_metrics['first_char_acc'])}`，blank-top1 均值 `{board_metrics['blank_top1_mean']:.4f}`"
        )
        for item in board_metrics['details']:
            lines.append(
                f"- 锚点样本：`{Path(item['image_path']).name}` gt=`{item['gt']}` pred=`{item['pred']}` top5={item['first_char_top5']}"
            )
    lines.append("")

    lines.append("## 泛化与过拟合判断")
    lines.append(f"- {summarize_generalization(val_metrics, test_metrics)}")
    if train_info.get('epoch_rows') and val_metrics and test_metrics:
        lines.append(
            "- 训练损失持续下降但测试集仍明显低于验证集时，更应优先怀疑部署域差、真实板图样本不足或训练分布偏置，而不是单纯增加 epoch。"
        )
    lines.append("")

    lines.append("## 产物")
    if (run_dir / 'weights' / 'Final_LPRNet_model.pth').exists():
        lines.append(f"- 最终权重：`{run_dir / 'weights' / 'Final_LPRNet_model.pth'}`")
    for name in ['LPRNet_stage3_rk3568_fp16.onnx', 'LPRNet_stage3_rk3568_fp16.rknn']:
        path = run_dir / 'weights' / name
        if path.exists():
            lines.append(f"- `{path}`")
    for name in ['val_metrics.json', 'test_metrics.json', 'board_anchor_metrics.json', 'train.log']:
        path = run_dir / name
        if path.exists():
            lines.append(f"- `{path}`")
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(report_path)


if __name__ == '__main__':
    main()

# Board-Aligned LPRNet Fine-Tune

This repo now supports a board-aligned CCPD training mode that mimics the ARM-side OCR preprocessing path:

- `ocr_channel_order=bgr`
- `ocr_crop_mode=match`
- `ocr_resize_mode=letterbox`
- `ocr_resize_kernel=nn`
- `ocr_preproc=none`
- `ocr_min_occ_ratio=0.90`

The preprocessing logic is implemented in [load_data.py](/home/wzzz/LPRNet/load_data.py) via `CCPDBoardDataLoader`.

## Train

```bash
conda activate /home/wzzz/LPRNet/.conda

python train_LPRNet.py \
  --train_img_dirs ./CCPD2019 \
  --test_img_dirs ./CCPD2019 \
  --train_txt_file ./prepared_labels/ccpd2019/train_labels.txt \
  --test_txt_file ./prepared_labels/ccpd2019/val_labels.txt \
  --data_mode ccpd_board \
  --ocr_channel_order bgr \
  --ocr_crop_mode match \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --pretrained_model ./weights_red_stage3/Final_LPRNet_model.pth \
  --save_folder ./weights_board_aligned/ \
  --cuda true
```

## Evaluate

```bash
python test_LPRNet.py \
  --test_img_dirs ./CCPD2019 \
  --txt_file ./prepared_labels/ccpd2019/test_labels.txt \
  --data_mode ccpd_board \
  --ocr_channel_order bgr \
  --ocr_crop_mode match \
  --ocr_resize_mode letterbox \
  --ocr_resize_kernel nn \
  --ocr_preproc none \
  --ocr_min_occ_ratio 0.90 \
  --pretrained_model ./weights_board_aligned/Final_LPRNet_model.pth \
  --cuda true
```

## Notes

- This mode crops from the full CCPD image using the bbox encoded in the filename.
- `match-ytrim` recrop is applied when occupancy is below `ocr_min_occ_ratio`, matching the board logic.
- If you want to experiment with board-side alternatives, only change the `--ocr_*` arguments here and keep the board runtime aligned.
- The first formal baseline experiment is automated by `./run_first_board_experiment.sh first_board_baseline_v1`.

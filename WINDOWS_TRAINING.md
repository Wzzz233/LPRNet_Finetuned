# Windows Training

## Purpose

Use Windows for PyTorch training and keep RKNN conversion in WSL.

## Current status

- Conda env is local to this folder: `.conda`
- Windows training uses PyTorch CUDA
- RKNN conversion should stay in your WSL `rknn_toolkit` environment

## Activate environment

```powershell
conda activate C:\Users\Wzzz2\OneDrive\Desktop\LPRNet\.conda
```

Or run without activation:

```powershell
conda run -p .\.conda python train_LPRNet.py --help
```

## Required data

This repo now supports the raw `CCPD2019` layout in this folder.

First generate label files from `CCPD2019/splits/*.txt`:

```powershell
conda run -p .\.conda python prepare_ccpd_splits.py `
  --dataset_root .\CCPD2019 `
  --output_dir .\prepared_labels\ccpd2019
```

This creates:

- `prepared_labels\ccpd2019\train_labels.txt`
- `prepared_labels\ccpd2019\val_labels.txt`
- `prepared_labels\ccpd2019\test_labels.txt`

Each line is `relative/path/to/image.jpg 车牌文本`.

## Recommended first training command

```powershell
.\run_train_windows.ps1 `
  -TrainDir ".\CCPD2019" `
  -TestDir ".\CCPD2019" `
  -TrainLabelFile ".\prepared_labels\ccpd2019\train_labels.txt" `
  -TestLabelFile ".\prepared_labels\ccpd2019\val_labels.txt" `
  -Cuda
```

## Manual training command

```powershell
conda run -p .\.conda python train_LPRNet.py `
  --train_img_dirs ".\CCPD2019" `
  --test_img_dirs ".\CCPD2019" `
  --train_txt_file ".\prepared_labels\ccpd2019\train_labels.txt" `
  --test_txt_file ".\prepared_labels\ccpd2019\val_labels.txt" `
  --pretrained_model ".\weights_red_stage3\Final_LPRNet_model.pth" `
  --save_folder ".\weights_local\" `
  --cuda true
```

## Notes

- `build_lprnet()` is fixed so `phase_train=True` now really means train mode
- `train_LPRNet.py` now supports separate `--train_txt_file` and `--test_txt_file`
- `load_data.py` now supports relative paths like `ccpd_base/xxx.jpg`
- default pretrained weights now point to `weights_red_stage3/Final_LPRNet_model.pth`
- local outputs go to `weights_local/`

## WSL side

After Windows training:

```powershell
conda run -p .\.conda python export_onnx.py --weights .\weights_local\Final_LPRNet_model.pth
```

Then move the ONNX into WSL and use your RKNN workflow there.

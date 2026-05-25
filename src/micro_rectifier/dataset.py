from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def build_record_from_crop_pair(sample_id, input_path, target_path, text, split, source_name, dx, dy, sx, sy, shx):
    return {
        'sample_id': str(sample_id),
        'input_path': str(input_path),
        'target_path': str(target_path),
        'text': str(text),
        'split': str(split),
        'source_name': str(source_name),
        'params': [float(dx), float(dy), float(sx), float(sy), float(shx)],
    }


def load_records(path_or_records):
    if isinstance(path_or_records, (str, Path)):
        raise NotImplementedError('jsonl loading will be added in implementation phase')
    return list(path_or_records)


class MicroRectifierDataset(Dataset):
    def __init__(self, records: Iterable[dict], input_size=(160, 48), grayscale: bool = False):
        self.records = load_records(records)
        self.input_w = int(input_size[0])
        self.input_h = int(input_size[1])
        self.grayscale = bool(grayscale)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        src = cv2.imread(str(row['input_path']), cv2.IMREAD_COLOR)
        tgt = cv2.imread(str(row['target_path']), cv2.IMREAD_COLOR)
        if src is None or tgt is None:
            raise FileNotFoundError(row)
        src = cv2.resize(src, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        tgt = cv2.resize(tgt, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        if self.grayscale:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)[..., None]
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)[..., None]
        src = src.astype(np.float32) / 255.0
        tgt = tgt.astype(np.float32) / 255.0
        src = np.transpose(src, (2, 0, 1))
        tgt = np.transpose(tgt, (2, 0, 1))
        return {
            'image': torch.from_numpy(src).float(),
            'target_image': torch.from_numpy(tgt).float(),
            'params': torch.tensor(row['params'], dtype=torch.float32),
            'sample_id': row['sample_id'],
            'text': row['text'],
            'source_name': row['source_name'],
        }

from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os
from dataclasses import dataclass

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
PROVINCE_COUNT = 31

OCR_CROP_WIDTH = 150
OCR_CROP_HEIGHT = 50


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self):
        return self.x2 - self.x1 + 1

    @property
    def h(self):
        return self.y2 - self.y1 + 1


def parse_ccpd_bbox_from_name(image_name):
    stem = os.path.splitext(os.path.basename(image_name))[0]
    parts = stem.split('-')
    if len(parts) < 3:
        return None
    bbox = parts[2]
    if '_' not in bbox:
        return None
    p1, p2 = bbox.split('_', 1)
    if '&' not in p1 or '&' not in p2:
        return None
    x1s, y1s = p1.split('&', 1)
    x2s, y2s = p2.split('&', 1)
    try:
        return Box(int(x1s), int(y1s), int(x2s), int(y2s))
    except ValueError:
        return None


def read_ppm_p6_payload(path):
    with open(path, 'rb') as f:
        blob = f.read()
    if not blob.startswith(b'P6'):
        raise ValueError(f'unsupported ppm format: {path}')
    i = 2
    tokens = []
    n = len(blob)
    while len(tokens) < 3:
        while i < n and blob[i] in b' \t\r\n':
            i += 1
        if i < n and blob[i] == ord('#'):
            while i < n and blob[i] not in b'\r\n':
                i += 1
            continue
        j = i
        while j < n and blob[j] not in b' \t\r\n':
            j += 1
        tokens.append(blob[i:j].decode('ascii'))
        i = j
    w, h, maxv = map(int, tokens)
    if maxv != 255:
        raise ValueError(f'unsupported ppm max value {maxv} in {path}')
    while i < n and blob[i] in b' \t\r\n':
        i += 1
    payload = np.frombuffer(blob[i:], dtype=np.uint8)
    if payload.size != w * h * 3:
        raise ValueError(f'invalid ppm payload size in {path}: expect {w*h*3}, got {payload.size}')
    return payload.reshape(h, w, 3).copy()


def load_label_items(txt_file):
    label_items = []
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    rel_path = os.path.normpath(parts[0].replace('\\', '/'))
                    filename = os.path.basename(rel_path)
                    label_items.append((rel_path, filename, parts[1]))
    else:
        print(f"【严重警告】找不到标签文件 {txt_file}")
    return label_items


def clamp_box(box, img_w, img_h):
    x1 = max(0, min(img_w - 1, box.x1))
    y1 = max(0, min(img_h - 1, box.y1))
    x2 = max(0, min(img_w - 1, box.x2))
    y2 = max(0, min(img_h - 1, box.y2))
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return Box(x1, y1, x2, y2)


def order_quad_points(pts):
    quad = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = quad.sum(axis=1)
    diffs = np.diff(quad, axis=1).reshape(-1)
    ordered[0] = quad[np.argmin(sums)]   # top-left
    ordered[2] = quad[np.argmax(sums)]   # bottom-right
    ordered[1] = quad[np.argmin(diffs)]  # top-right
    ordered[3] = quad[np.argmax(diffs)]  # bottom-left
    return ordered


def quad_to_box(pts, img_w=None, img_h=None):
    quad = order_quad_points(pts)
    x1 = int(np.floor(np.min(quad[:, 0])))
    y1 = int(np.floor(np.min(quad[:, 1])))
    x2 = int(np.ceil(np.max(quad[:, 0])))
    y2 = int(np.ceil(np.max(quad[:, 1])))
    box = Box(x1, y1, x2, y2)
    if img_w is not None and img_h is not None:
        box = clamp_box(box, img_w, img_h)
    return box


def expand_quad(pts, pad_ratio):
    quad = order_quad_points(pts)
    if pad_ratio <= 0.0:
        return quad
    center = np.mean(quad, axis=0, keepdims=True)
    scale = 1.0 + float(pad_ratio)
    return (center + (quad - center) * scale).astype(np.float32)


def clip_quad_to_image(pts, img_w, img_h):
    quad = np.asarray(pts, dtype=np.float32).reshape(4, 2).copy()
    quad[:, 0] = np.clip(quad[:, 0], 0, max(0, img_w - 1))
    quad[:, 1] = np.clip(quad[:, 1], 0, max(0, img_h - 1))
    return order_quad_points(quad)


def quad_edge_lengths(pts):
    quad = order_quad_points(pts)
    width_top = float(np.linalg.norm(quad[1] - quad[0]))
    width_bottom = float(np.linalg.norm(quad[2] - quad[3]))
    height_left = float(np.linalg.norm(quad[3] - quad[0]))
    height_right = float(np.linalg.norm(quad[2] - quad[1]))
    return width_top, width_bottom, height_left, height_right


def warp_quad_to_rect(image, pts, dst_w=None, dst_h=None, pad_ratio=0.0):
    img_h, img_w = image.shape[:2]
    quad = expand_quad(pts, pad_ratio)
    quad = clip_quad_to_image(quad, img_w, img_h)
    width_top, width_bottom, height_left, height_right = quad_edge_lengths(quad)
    if dst_w is None:
        dst_w = int(max(width_top, width_bottom) + 0.5)
    if dst_h is None:
        dst_h = int(max(height_left, height_right) + 0.5)
    dst_w = max(1, int(dst_w))
    dst_h = max(1, int(dst_h))
    dst = np.array(
        [
            [0.0, 0.0],
            [dst_w - 1.0, 0.0],
            [dst_w - 1.0, dst_h - 1.0],
            [0.0, dst_h - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, matrix, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
    return warped, quad, matrix


def compute_center_crop_box(src, img_w, img_h, crop_w, crop_h):
    cx = (src.x1 + src.x2) // 2
    cy = (src.y1 + src.y2) // 2
    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x1 + crop_w > img_w:
        x1 = img_w - crop_w
    if y1 + crop_h > img_h:
        y1 = img_h - crop_h
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    return clamp_box(Box(x1, y1, x1 + crop_w - 1, y1 + crop_h - 1), img_w, img_h)


def compute_expand_crop_box(src, img_w, img_h, pad_x, pad_y):
    ex = int(src.w * pad_x + 0.5)
    ey = int(src.h * pad_y + 0.5)
    return clamp_box(Box(src.x1 - ex, src.y1 - ey, src.x2 + ex, src.y2 + ey), img_w, img_h)


def compute_ocr_crop_box(src, img_w, img_h, crop_mode):
    if crop_mode == 'match':
        return clamp_box(src, img_w, img_h)
    if crop_mode == 'box':
        return compute_expand_crop_box(src, img_w, img_h, 0.06, 0.12)
    if crop_mode == 'tight':
        return compute_expand_crop_box(src, img_w, img_h, 0.08, 0.16)
    if crop_mode == 'box-pad':
        return compute_expand_crop_box(src, img_w, img_h, 0.15, 0.28)
    return compute_center_crop_box(src, img_w, img_h, OCR_CROP_WIDTH, OCR_CROP_HEIGHT)


def estimate_ocr_occ_ratio(crop_w, crop_h, in_w, in_h, resize_mode):
    if crop_w <= 0 or crop_h <= 0 or in_w <= 0 or in_h <= 0:
        return 0.0
    if resize_mode != 'letterbox':
        return 1.0
    sx = in_w / float(crop_w)
    sy = in_h / float(crop_h)
    scale = min(sx, sy)
    if scale <= 0.0:
        return 0.0
    scaled_w = int(crop_w * scale + 0.5)
    scaled_w = max(1, min(in_w, scaled_w))
    return scaled_w / float(in_w)


def compute_match_ytrim_crop(src, in_w, in_h, min_occ_ratio):
    if min_occ_ratio <= 0.0:
        return None
    bw = src.w
    bh = src.h
    if bw <= 0 or bh <= 0 or in_w <= 0 or in_h <= 0:
        return None
    min_occ_ratio = min(1.0, min_occ_ratio)
    model_aspect = in_w / float(in_h)
    target_aspect = model_aspect * min_occ_ratio
    if target_aspect <= 0.0:
        return None
    cur_aspect = bw / float(bh)
    if cur_aspect >= target_aspect:
        return None
    new_h = int(bw / target_aspect + 0.5)
    if new_h < 1:
        new_h = 1
    if new_h >= bh:
        return None
    trim = (bh - new_h) // 2
    y1 = src.y1 + trim
    y2 = y1 + new_h - 1
    if y2 > src.y2:
        y2 = src.y2
        y1 = y2 - new_h + 1
    if y1 < src.y1:
        y1 = src.y1
    if y2 <= y1:
        return None
    return Box(src.x1, y1, src.x2, y2)


def box_iou(a, b):
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = float((ix2 - ix1 + 1) * (iy2 - iy1 + 1))
    area_a = float(a.w * a.h)
    area_b = float(b.w * b.h)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def simulate_refined_plate_box(src, img_w, img_h, jitter_x, jitter_y, min_iou, max_tries=8):
    bw = src.w
    bh = src.h
    if bw <= 1 or bh <= 1:
        return src

    max_dx = max(1, int(bw * max(0.0, jitter_x) + 0.5))
    max_dy = max(1, int(bh * max(0.0, jitter_y) + 0.5))
    min_w = max(4, int(bw * 0.60 + 0.5))
    min_h = max(4, int(bh * 0.60 + 0.5))
    min_iou = max(0.0, min(1.0, min_iou))

    for _ in range(max_tries):
        cand = Box(
            src.x1 + random.randint(-max_dx, max_dx),
            src.y1 + random.randint(-max_dy, max_dy),
            src.x2 + random.randint(-max_dx, max_dx),
            src.y2 + random.randint(-max_dy, max_dy),
        )
        cand = clamp_box(cand, img_w, img_h)
        if cand.w < min_w or cand.h < min_h:
            continue
        if box_iou(src, cand) >= min_iou:
            return cand
    return src


def resize_bgr_with_kernel(src, dst_w, dst_h, kernel):
    interp = cv2.INTER_LINEAR if kernel == 'bilinear' else cv2.INTER_NEAREST
    return cv2.resize(src, (dst_w, dst_h), interpolation=interp)


def resize_bgr_letterbox(src, dst_w, dst_h, kernel, pad_value=0):
    src_h, src_w = src.shape[:2]
    out = np.full((dst_h, dst_w, 3), pad_value, dtype=np.uint8)
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    scale = min(sx, sy)
    if scale <= 0.0:
        return out, 0.0
    scaled_w = int(src_w * scale + 0.5)
    scaled_h = int(src_h * scale + 0.5)
    scaled_w = max(1, min(dst_w, scaled_w))
    scaled_h = max(1, min(dst_h, scaled_h))
    off_x = (dst_w - scaled_w) // 2
    off_y = (dst_h - scaled_h) // 2
    resized = resize_bgr_with_kernel(src, scaled_w, scaled_h, kernel)
    out[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = resized
    return out, scale


def ocr_preprocess_bgr888(img, mode):
    mode = (mode or 'none').strip().lower()
    if mode == 'raw':
        mode = 'none'
    if mode == 'gray3':
        mode = 'gray'
    if mode == 'none':
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == 'gray':
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == 'bin':
        tmp = cv2.blur(gray, (3, 3))
        thr = cv2.blur(tmp, (5, 5))
        out = np.where(tmp > (thr - 8), 255, 0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    raise ValueError(f'unsupported ocr preproc mode: {mode}')


def prepare_board_ocr_input_bgr888(crop_bgr, in_w, in_h, resize_mode, resize_kernel, preproc_mode, channel_order):
    work = ocr_preprocess_bgr888(crop_bgr.copy(), preproc_mode)
    if resize_mode == 'letterbox':
        out, scale = resize_bgr_letterbox(work, in_w, in_h, resize_kernel, pad_value=0)
        scaled_w = int(work.shape[1] * scale + 0.5)
        scaled_w = max(1, min(in_w, scaled_w))
        occ = scaled_w / float(in_w)
    else:
        out = resize_bgr_with_kernel(work, in_w, in_h, resize_kernel)
        occ = 1.0
    if channel_order == 'rgb':
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out, occ


def prepare_board_ocr_input_from_quad_bgr888(
    image,
    quad,
    in_w,
    in_h,
    resize_mode,
    resize_kernel,
    preproc_mode,
    channel_order,
    quad_pad_ratio=0.0,
):
    warped, ordered_quad, matrix = warp_quad_to_rect(image, quad, pad_ratio=quad_pad_ratio)
    prepared, occ = prepare_board_ocr_input_bgr888(
        warped,
        in_w,
        in_h,
        resize_mode,
        resize_kernel,
        preproc_mode,
        channel_order,
    )
    return prepared, occ, warped, ordered_quad, matrix

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None, txt_file='./ocrin_ccpd/train_labels.txt'):
        self.img_dir = img_dir
        self.img_paths = []
        self.img_labels = [] # 新增：专门用来存真实的汉字标签
        self.img_rel_paths = []

        # --- 核心修改：读取我们的标准答案 txt 文件 ---
        label_items = load_label_items(txt_file)
        if not label_items:
            print(f"【严重警告】标签文件为空或无有效记录 {txt_file}，请确保它和运行代码在同一文件夹！")

        # 遍历文件夹，把存在的图片和标签配对。
        # 优先支持带子目录的相对路径（例如 CCPD2019/splits/train.txt 生成的 ccpd_base/xxx.jpg），
        # 同时保留旧格式 basename-only 标签文件的兼容性。
        seen_paths = set()
        for img_folder in img_dir: 
            root_dir = os.path.normpath(img_folder)
            for rel_path, filename, label in label_items:
                candidate_paths = [
                    os.path.join(root_dir, rel_path),
                    os.path.join(root_dir, filename),
                ]
                for full_path in candidate_paths:
                    norm_full_path = os.path.normpath(full_path)
                    if os.path.exists(norm_full_path) and norm_full_path not in seen_paths:
                        self.img_paths.append(norm_full_path)
                        self.img_labels.append(label)
                        self.img_rel_paths.append(rel_path)
                        seen_paths.add(norm_full_path)
                        break

        # 把图片路径和标签绑在一起打乱（洗牌），防止模型死记硬背
        combined = list(zip(self.img_paths, self.img_labels, self.img_rel_paths))
        random.shuffle(combined)
        if combined:
            self.img_paths, self.img_labels, self.img_rel_paths = zip(*combined)
            self.img_paths = list(self.img_paths)
            self.img_labels = list(self.img_labels)
            self.img_rel_paths = list(self.img_rel_paths)

        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        text_label = self.img_labels[index] # 直接拿到我们 txt 里的真实车牌号！

        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        # 强制将所有图片缩放成 94x24，满足 LPRNet 的胃口
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        label = list()
        for c in text_label:
            # 去字典里查这个字对应的数字密码，存给模型
            label.append(CHARS_DICT[c])

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def check(self, label):
        # 原版有一些严格的新能源车牌格式校验（必须第3位是D/F之类）
        # 我们的车牌来自 CCPD，格式多样，为了防止训练中断，直接放行
        return True


class CCPDBoardDataLoader(Dataset):
    def __init__(
        self,
        img_dir,
        imgSize,
        lpr_max_len,
        txt_file,
        ocr_channel_order='bgr',
        ocr_crop_mode='match',
        ocr_resize_mode='letterbox',
        ocr_resize_kernel='nn',
        ocr_preproc='none',
        ocr_min_occ_ratio=0.90,
        plate_box_aug_mode='none',
        plate_box_aug_prob=0.0,
        plate_box_aug_x=0.06,
        plate_box_aug_y=0.12,
        plate_box_aug_min_iou=0.75,
    ):
        self.img_paths = []
        self.img_labels = []
        self.img_rel_paths = []
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.ocr_channel_order = ocr_channel_order
        self.ocr_crop_mode = ocr_crop_mode
        self.ocr_resize_mode = ocr_resize_mode
        self.ocr_resize_kernel = ocr_resize_kernel
        self.ocr_preproc = ocr_preproc
        self.ocr_min_occ_ratio = ocr_min_occ_ratio
        self.plate_box_aug_mode = plate_box_aug_mode
        self.plate_box_aug_prob = plate_box_aug_prob
        self.plate_box_aug_x = plate_box_aug_x
        self.plate_box_aug_y = plate_box_aug_y
        self.plate_box_aug_min_iou = plate_box_aug_min_iou

        label_items = load_label_items(txt_file)

        seen_paths = set()
        for img_folder in img_dir:
            root_dir = os.path.normpath(img_folder)
            for rel_path, filename, label in label_items:
                candidate_paths = [
                    os.path.join(root_dir, rel_path),
                    os.path.join(root_dir, filename),
                ]
                for full_path in candidate_paths:
                    norm_full_path = os.path.normpath(full_path)
                    if os.path.exists(norm_full_path) and norm_full_path not in seen_paths:
                        self.img_paths.append(norm_full_path)
                        self.img_labels.append(label)
                        self.img_rel_paths.append(rel_path)
                        seen_paths.add(norm_full_path)
                        break

        combined = list(zip(self.img_paths, self.img_labels, self.img_rel_paths))
        random.shuffle(combined)
        if combined:
            self.img_paths, self.img_labels, self.img_rel_paths = zip(*combined)
            self.img_paths = list(self.img_paths)
            self.img_labels = list(self.img_labels)
            self.img_rel_paths = list(self.img_rel_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        text_label = self.img_labels[index]
        rel_path = self.img_rel_paths[index]

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f'Failed to read image: {img_path}')
        img_h, img_w = image.shape[:2]

        bbox = parse_ccpd_bbox_from_name(rel_path)
        if bbox is None:
            bbox = parse_ccpd_bbox_from_name(img_path)
        if bbox is None:
            raise RuntimeError(f'Cannot parse CCPD bbox from path: {rel_path}')

        bbox = clamp_box(bbox, img_w, img_h)
        if self.plate_box_aug_mode != 'none' and self.plate_box_aug_prob > 0.0:
            if random.random() < self.plate_box_aug_prob:
                if self.plate_box_aug_mode == 'jitter_refine':
                    bbox = simulate_refined_plate_box(
                        bbox,
                        img_w,
                        img_h,
                        self.plate_box_aug_x,
                        self.plate_box_aug_y,
                        self.plate_box_aug_min_iou,
                    )
        crop_box = compute_ocr_crop_box(bbox, img_w, img_h, self.ocr_crop_mode)
        occ_ratio = estimate_ocr_occ_ratio(crop_box.w, crop_box.h, self.img_size[0], self.img_size[1], self.ocr_resize_mode)
        if self.ocr_min_occ_ratio > 0.0 and occ_ratio < self.ocr_min_occ_ratio and self.ocr_crop_mode != 'tight':
            if self.ocr_crop_mode == 'match':
                recrop = compute_match_ytrim_crop(crop_box, self.img_size[0], self.img_size[1], self.ocr_min_occ_ratio)
                if recrop is not None:
                    crop_box = clamp_box(recrop, img_w, img_h)
            else:
                crop_box = compute_expand_crop_box(bbox, img_w, img_h, 0.08, 0.16)

        crop_bgr = image[crop_box.y1:crop_box.y2 + 1, crop_box.x1:crop_box.x2 + 1]
        crop_bgr, _ = prepare_board_ocr_input_bgr888(
            crop_bgr,
            self.img_size[0],
            self.img_size[1],
            self.ocr_resize_mode,
            self.ocr_resize_kernel,
            self.ocr_preproc,
            self.ocr_channel_order,
        )

        crop_bgr = crop_bgr.astype('float32')
        crop_bgr -= 127.5
        crop_bgr *= 0.0078125
        crop_bgr = np.transpose(crop_bgr, (2, 0, 1))

        label = []
        for c in text_label:
            label.append(CHARS_DICT[c])
        return crop_bgr, label, len(label)


class BoardDumpDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, txt_file):
        self.img_paths = []
        self.img_labels = []
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len

        label_items = load_label_items(txt_file)
        seen_paths = set()
        for img_folder in img_dir:
            root_dir = os.path.normpath(img_folder)
            for rel_path, filename, label in label_items:
                candidate_paths = [
                    os.path.join(root_dir, rel_path),
                    os.path.join(root_dir, filename),
                ]
                for full_path in candidate_paths:
                    norm_full_path = os.path.normpath(full_path)
                    if os.path.exists(norm_full_path) and norm_full_path not in seen_paths:
                        self.img_paths.append(norm_full_path)
                        self.img_labels.append(label)
                        seen_paths.add(norm_full_path)
                        break

        combined = list(zip(self.img_paths, self.img_labels))
        random.shuffle(combined)
        if combined:
            self.img_paths, self.img_labels = zip(*combined)
            self.img_paths = list(self.img_paths)
            self.img_labels = list(self.img_labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        text_label = self.img_labels[index]
        image = read_ppm_p6_payload(img_path)
        img_h, img_w = image.shape[:2]
        if img_w != self.img_size[0] or img_h != self.img_size[1]:
            raise RuntimeError(
                f'Board dump size mismatch for {img_path}: got {(img_w, img_h)} expect {tuple(self.img_size)}'
            )

        image = image.astype('float32')
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))

        label = []
        for c in text_label:
            label.append(CHARS_DICT[c])
        return image, label, len(label)

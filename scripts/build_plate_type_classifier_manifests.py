#!/usr/bin/env python3
"""
Build 6-class plate type classifier manifests.
Classes: blue=0, green=1, yellow=2, police=3, embassy=4, other=5

Sources:
  CCPD2019  → blue
  CCPD2020  → green
  CRPD_all  → blue (main), some police/school
  special_v2 → police, embassy
  git_plate → mixed, supplement
  CBLPRD    → mixed, GAN supplement (weight=0.2)

Blacklist: CRPD_raw_ccpd_board_v1 (zero files)
"""
import os, sys, csv, json, hashlib, math, random
from collections import defaultdict, Counter
from pathlib import Path
import cv2
import numpy as np

random.seed(20260602)

LPRNET_ROOT = Path("/home/wzzz/LPRNet")
OUT_DIR = LPRNET_ROOT / "manifests_rebased" / "plate_type_classifier_6cls_20260602"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {"blue": 0, "green": 1, "yellow": 2, "police": 3, "embassy": 4, "other": 5}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

MANIFEST_FIELDS = [
    "img_path", "label", "label_name", "plate_text",
    "source", "source_split", "crop_mode",
    "original_w", "original_h",
    "is_synthetic", "is_gan", "base_id",
    "sample_weight", "notes",
]

# ── Helpers ──────────────────────────────────────────────────────
def classify_6cls(text: str, source_hint: str = "") -> str:
    """Classify plate text into 6 classes."""
    if not text:
        return "other"
    t = text.strip()
    # embassy ONLY for 使-starting plates
    if t.startswith("使"):
        return "embassy"
    # 领 → other, never embassy
    if t.startswith("领"):
        return "other"
    # police
    if t.endswith("警"):
        return "police"
    # 学/挂 → other
    if t.endswith("学") or t.endswith("挂"):
        return "other"
    # 港/澳 → other
    if "港" in t or "澳" in t:
        return "other"
    # 8-char → green
    if len(t) == 8:
        return "green"
    # HK/MO in body → other (cross-border)
    if "HK" in t or "MO" in t:
        return "other"
    # 7-char → blue (default)
    if len(t) == 7:
        return "blue"
    # Other lengths → other (but check for yellow by length)
    # CBLPRD 双层黄牌 is 7-char, handled above
    return "other"


def make_base_id(img_path: str) -> str:
    """Make a cross-source unique base ID from image path."""
    p = Path(img_path)
    stem = p.stem
    # Use stem + parent suffix for uniqueness
    parent = p.parent.name if p.parent.name != "." else ""
    raw = f"{parent}__{stem}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def img_size(img_path) -> tuple:
    """Get image dimensions from header (fast, no full decode)."""
    from PIL import Image
    try:
        with Image.open(str(img_path)) as im:
            return im.size  # w, h
    except:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                return img.shape[1], img.shape[0]
        except:
            pass
    return (0, 0)


def ensure_img_exists(img_path) -> bool:
    """Check image file exists."""
    return Path(img_path).is_file()


def source_weight(source: str, label_name: str) -> float:
    """Assign sample weight based on source and class."""
    w = {
        "ccpd2019": 1.0,
        "ccpd2020": 1.5,  # scarce real green
        "crpd_single": 1.0,
        "crpd_double": 1.0,
        "crpd_multi": 1.0,
        "special_v2": 1.0,
        "git_plate": 0.5,
        "cblprd": 0.2,
    }.get(source, 1.0)
    # Heavier weight for rare classes from good sources
    if source == "special_v2" and label_name in ("police", "embassy"):
        w = 2.0  # primary training data for these classes
    # git_plate police/embassy: only used as hard negative, very low weight
    if source == "git_plate" and label_name in ("police", "embassy"):
        w = 0.1
    return w


# ── Record accumulator ──────────────────────────────────────────
records = []  # list of dicts

def add_record(img_path, label_name, plate_text, source, source_split,
               crop_mode, w, h, is_synthetic=False, is_gan=False,
               base_id=None, notes=""):
    """Add a manifest record."""
    if not ensure_img_exists(img_path):
        return
    if base_id is None:
        base_id = make_base_id(str(img_path))
    label = CLASS_MAP.get(label_name, 5)  # default other
    sw = source_weight(source, label_name)
    records.append({
        "img_path": str(img_path),
        "label": label,
        "label_name": label_name,
        "plate_text": plate_text,
        "source": source,
        "source_split": source_split,
        "crop_mode": crop_mode,
        "original_w": w,
        "original_h": h,
        "is_synthetic": int(is_synthetic),
        "is_gan": int(is_gan),
        "base_id": base_id,
        "sample_weight": round(sw, 2),
        "notes": notes,
    })


# ── 1. CCPD2019 → blue ──────────────────────────────────────────
def scan_ccpd2019():
    """Use existing posquad manifest for blue plates. Sample for manifest building."""
    manifest = LPRNET_ROOT / "manifests_rebased" / "blue_ccpd2019_tilt_db_challenge_posquad_20260508" / "all_posquad.csv"
    if not manifest.exists():
        print("  [SKIP] CCPD2019 manifest not found")
        return
    count = 0
    with open(manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            count += 1
            if count > 30000:
                break
            img_path = LPRNET_ROOT / row["img_path"]
            if not img_path.exists():
                continue
            text = row.get("text", "")
            if not text or classify_6cls(text) != "blue":
                continue
            w, h = img_size(img_path)
            if w == 0:
                continue
            base_id = make_base_id(str(img_path))
            split = row.get("split", "train")
            add_record(img_path, "blue", text, "ccpd2019", split,
                       "perspective_warp", w, h, base_id=base_id)


# ── 2. CCPD2020 → green ─────────────────────────────────────────
def scan_ccpd2020():
    manifest = LPRNET_ROOT / "manifests_rebased" / "ccpd2020_green_real_20260509" / "train_ccpd2020_green_real.csv"
    if not manifest.exists():
        print("  [SKIP] CCPD2020 manifest not found")
        return
    with open(manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = LPRNET_ROOT / row["img_path"]
            if not img_path.exists():
                continue
            text = row.get("text", "")
            if not text or classify_6cls(text) != "green":
                continue
            w, h = img_size(img_path)
            if w == 0:
                continue
            base_id = make_base_id(str(img_path))
            add_record(img_path, "green", text, "ccpd2020", "train",
                       "perspective_warp", w, h, base_id=base_id)
    # Also add raw files for val splits
    green_dir = LPRNET_ROOT / "datasets" / "CCPD2020" / "ccpd_green"
    for split in ["val", "test"]:
        d = green_dir / split
        if not d.is_dir():
            continue
        for fname in os.listdir(d):
            if not fname.endswith(".jpg"):
                continue
            img_path = d / fname
            # Extract text from CCPD filename
            parts = fname.split("-")
            if len(parts) >= 4:
                text_part = parts[-1].replace(".jpg", "")
                # CCPD2020 text is numeric-encoded; use existing label CSV
                continue  # skip raw files without text parsing
            # Actually use the existing manifest for val/test too
            pass


# ── 3. CRPD_all → mostly blue, some other ───────────────────────
def scan_crpd():
    base = LPRNET_ROOT / "datasets" / "CRPD_all"
    for layout in ["CRPD_single", "CRPD_double", "CRPD_multi"]:
        src = f"crpd_{layout.split('_')[-1].lower()}"
        count = 0
        for split in ["train", "val", "test"]:
            img_dir = base / layout / split / "images"
            lbl_dir = base / layout / split / "labels"
            if not img_dir.is_dir():
                continue
            for fname in sorted(os.listdir(img_dir)):
                if not fname.endswith(".jpg"):
                    continue
                count += 1
                if count > 1500:
                    break
                img_path = img_dir / fname
                lbl_path = lbl_dir / (fname.rsplit(".", 1)[0] + ".txt")
                if not lbl_path.exists():
                    continue
                # Get image dimensions without full decode
                try:
                    w, h = img_size(img_path)
                    if w == 0:
                        continue
                except:
                    continue
                with open(lbl_path) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) < 9:
                            continue
                        text = parts[-1]
                        if not text:
                            continue
                        cls = classify_6cls(text)
                        if cls not in ("blue", "police", "other"):
                            cls = "other"
                        bi = make_base_id(f"{fname}_{parts[0]}_{parts[4]}")
                        add_record(img_path, cls, text, src, split,
                                   "perspective_warp", w, h, base_id=bi)


# ── 4. special_v2 → police, embassy ─────────────────────────────
def scan_special_v2():
    for split in ["train", "val_clean", "val_hard"]:
        for cls_label, family in [("police", "police"), ("embassy", "embassy")]:
            manifest = LPRNET_ROOT / "manifests_rebased" / "special_split_v2_20260601" / f"{split}_{family}.csv"
            if not manifest.exists():
                continue
            msrc = f"special_v2_{split}"
            with open(manifest) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "")
                    ipath = row.get("img_path", "")
                    if not text or not ipath:
                        continue
                    # Resolve path
                    if "scripts/special_gen/datasets/" in ipath:
                        rel = ipath.split("scripts/special_gen/datasets/", 1)[1]
                    else:
                        rel = ipath
                    p = LPRNET_ROOT / "datasets" / rel
                    if not p.exists():
                        continue
                    w, h = img_size(p)
                    if w == 0:
                        continue
                    bi = make_base_id(str(p))
                    add_record(p, cls_label, text, "special_v2", split,
                               "perspective_warp", w, h, is_synthetic=True,
                               base_id=bi)


# ── 5. git_plate → supplement ───────────────────────────────────
def scan_git_plate():
    import re
    base = LPRNET_ROOT / "datasets" / "git_plate" / "train"
    chinese_start = re.compile(r"^[\u4e00-\u9fff使领]")
    cls_counts = Counter()
    count = 0

    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".jpg"):
            continue
        if "_distort" in fname or "_stretch" in fname:
            continue
        count += 1
        if count > 20000:
            break

        # Extract plate text
        stem = fname.rsplit(".", 1)[0]
        parts = stem.split("_")
        text_parts = [p for p in parts if chinese_start.match(p)]
        text = text_parts[-1] if text_parts else stem

        cls = classify_6cls(text, "git_plate")

        img_path = base / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        bi = make_base_id(f"git_{text}_{parts[-1] if parts else stem}")

        # git_plate police/embassy → val_cross_source only
        if cls in ("police", "embassy"):
            add_record(img_path, cls, text, "git_plate", "val_cross_source",
                       "resize_pad", w, h, base_id=bi,
                       notes="pre_applied_warping_hard_negative")
            continue

        # Other classes: split to train
        # Use base_id hash for consistency
        hash_val = int(bi[:8], 16)
        split = "val_cross_source" if hash_val % 20 == 0 else "train"
        add_record(img_path, cls, text, "git_plate", split,
                   "resize_pad", w, h, base_id=bi)


# ── 6. CBLPRD → supplement (weight=0.2) ─────────────────────────
def scan_cblprd():
    df = LPRNET_ROOT / "datasets" / "CBLPRD-330k_v1" / "data.txt"
    idir = LPRNET_ROOT / "datasets" / "CBLPRD-330k_v1" / "CBLPRD-330k"
    if not df.exists():
        return

    type_to_class = {
        "普通蓝牌": "blue",
        "新能源小型车": "green",
        "新能源大型车": "green",
        "单层黄牌": "yellow",
        "双层黄牌": "yellow",
        "黑色车牌": "other",  # 黑牌→other (含使/领的由 text 决定)
        "拖拉机绿牌": "green",
    }

    cls_counts = Counter()
    count = 0
    with open(df) as f:
        for line in f:
            count += 1
            if count > 50000:
                break
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            fname = parts[0].split("/")[-1]
            text = parts[1]
            ptype = parts[2]

            # Check for special chars first
            if text.endswith("使"):
                cls = "embassy"
            elif text.endswith("领"):
                cls = "other"  # 领→other
            elif text.endswith("学") or text.endswith("挂"):
                cls = "other"
            elif text.startswith("使"):
                cls = "embassy"
            elif text.startswith("领"):
                cls = "other"
            elif "港" in text or "澳" in text:
                cls = "other"
            else:
                cls = type_to_class.get(ptype, "other")

            img_path = idir / fname
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            bi = make_base_id(f"cblprd_{fname}")

            # CBLPRD → only train, sample_weight=0.2 in field
            add_record(img_path, cls, text, "cblprd", "train",
                       "resize_pad", w, h, is_synthetic=True, is_gan=True,
                       base_id=bi)


# ── Split assignment ─────────────────────────────────────────────
def assign_splits():
    """Assign final split based on source_split and base_id hashing."""
    train = []
    val_clean = []
    val_hard = []
    val_cross_source = []
    final_holdout = []

    seen_train_ids = set()
    seen_val_ids = set()

    for r in records:
        src = r["source"]
        ssplit = r["source_split"]
        label = r["label_name"]
        bid = r["base_id"]

        # special_v2: use original split directly
        if src == "special_v2":
            if ssplit == "train":
                train.append(r)
                seen_train_ids.add(bid)
            elif ssplit == "val_clean":
                val_clean.append(r)
                seen_val_ids.add(bid)
            elif ssplit == "val_hard":
                val_hard.append(r)
                seen_val_ids.add(bid)
            continue

        # git_plate: val_cross_source already assigned
        if src == "git_plate" and ssplit == "val_cross_source":
            val_cross_source.append(r)
            continue

        # CRPD: use original split
        if src.startswith("crpd_"):
            if ssplit == "train":
                train.append(r)
                seen_train_ids.add(bid)
            elif ssplit == "val":
                val_clean.append(r)
                seen_val_ids.add(bid)
            elif ssplit == "test":
                val_hard.append(r)
                seen_val_ids.add(bid)
            continue

        # CBLPRD: split for class coverage
        if src == "cblprd":
            hash_val = int(bid[:8], 16)
            bucket = hash_val % 20
            if bucket >= 19:  # 5% to val_clean
                val_clean.append(r)
                seen_val_ids.add(bid)
            elif bucket >= 17:  # 10% to val_hard
                val_hard.append(r)
                seen_val_ids.add(bid)
            else:  # 85% train
                train.append(r)
                seen_train_ids.add(bid)
            continue

        # CCPD2019/CCPD2020: hash-based split
        if src in ("ccpd2019", "ccpd2020"):
            hash_val = int(bid[:8], 16)
            bucket = hash_val % 20

            # Avoid base_id overlap between splits
            if bid in seen_val_ids and bucket < 16:
                # was already assigned to val, keep in val
                val_clean.append(r)
                continue
            if bid in seen_train_ids:
                # skip duplicate
                continue

            if bucket < 14:  # 70% train
                train.append(r)
                seen_train_ids.add(bid)
            elif bucket < 16:  # 10% val_clean
                val_clean.append(r)
                seen_val_ids.add(bid)
            elif bucket < 18:  # 10% val_hard
                val_hard.append(r)
                seen_val_ids.add(bid)
            else:  # 10% cross-source
                val_cross_source.append(r)
            continue

        # git_plate train: hash-based split  
        if src == "git_plate" and ssplit == "train":
            hash_val = int(bid[:8], 16)
            bucket = hash_val % 20
            if bucket < 15:  # 75% train
                train.append(r)
                seen_train_ids.add(bid)
            elif bucket < 17:  # 10% val_clean
                val_clean.append(r)
                seen_val_ids.add(bid)
            elif bucket < 19:  # 10% val_hard
                val_hard.append(r)
                seen_val_ids.add(bid)
            else:  # 5% cross-source
                val_cross_source.append(r)
            continue

        # default: train
        train.append(r)

    # De-duplicate: if base_id appears in both train and val, remove from val
    train_ids = set(r["base_id"] for r in train)
    for split_list in [val_clean, val_hard, val_cross_source]:
        ids_before = len(split_list)
        split_list[:] = [r for r in split_list if r["base_id"] not in train_ids]
        removed = ids_before - len(split_list)
        if removed:
            print(f"  Removed {removed} val entries with base_id overlap with train")

    return train, val_clean, val_hard, val_cross_source, final_holdout


# ── Write outputs ────────────────────────────────────────────────
def write_csv(rows, name):
    path = OUT_DIR / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # Also write sample_weight-preserved version
    print(f"  {name}: {len(rows)} rows")
    return len(rows)


def write_class_map():
    path = OUT_DIR / "class_map.json"
    with open(path, "w") as f:
        json.dump(CLASS_MAP, f, indent=2)
    print(f"  class_map.json: {len(CLASS_MAP)} classes")


def write_source_stats(train, val_c, val_h, val_cs, holdout):
    """Write per-source, per-class statistics."""
    from collections import Counter
    stats = {}
    for split_name, split_data in [
        ("train", train), ("val_clean", val_c), ("val_hard", val_h),
        ("val_cross_source", val_cs), ("final_holdout", holdout)
    ]:
        by_source = defaultdict(Counter)
        by_class = Counter()
        for r in split_data:
            by_source[r["source"]][r["label_name"]] += 1
            by_class[r["label_name"]] += 1
        stats[split_name] = {
            "total": len(split_data),
            "by_class": dict(by_class),
            "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
        }
    path = OUT_DIR / "source_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  source_stats.json written")


def write_build_summary(train, val_c, val_h, val_cs, holdout):
    total = len(train) + len(val_c) + len(val_h) + len(val_cs) + len(holdout)
    summary = {
        "date": "2026-06-02",
        "class_map": CLASS_MAP,
        "total_records": total,
        "splits": {
            "train": len(train),
            "val_clean": len(val_c),
            "val_hard": len(val_h),
            "val_cross_source": len(val_cs),
            "final_holdout": len(holdout),
        },
        "sources_used": sorted(set(r["source"] for r in records)),
    }
    path = OUT_DIR / "build_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  build_summary.json written")
    return summary


def write_final_holdout_template():
    """Write empty template for final_holdout."""
    path = OUT_DIR / "final_holdout_template.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_FIELDS)
        w.writerow(["PLACEHOLDER"] + [""] * (len(MANIFEST_FIELDS) - 1))
    print(f"  final_holdout_template.csv written")


# ── Audit report ─────────────────────────────────────────────────
def print_audit(train, val_c, val_h, val_cs, holdout):
    print("\n" + "=" * 60)
    print("MANIFEST AUDIT")
    print("=" * 60)

    for name, data in [
        ("train", train), ("val_clean", val_c), ("val_hard", val_h),
        ("val_cross_source", val_cs), ("final_holdout", holdout)
    ]:
        print(f"\n  {name}: {len(data)} rows")
        by_class = Counter(r["label_name"] for r in data)
        print(f"    by class: {dict(by_class)}")
        by_source = Counter(r["source"] for r in data)
        print(f"    by source: {dict(by_source)}")

    # Base ID overlap check
    print("\n  Base ID overlap check:")
    splits = {"train": train, "val_clean": val_c, "val_hard": val_h,
              "val_cross_source": val_cs}
    ids_by_split = {k: set(r["base_id"] for r in v) for k, v in splits.items()}
    for s1 in splits:
        for s2 in splits:
            if s1 < s2:
                overlap = ids_by_split[s1] & ids_by_split[s2]
                if overlap:
                    print(f"    WARNING: {s1} ↔ {s2}: {len(overlap)} overlapping base IDs")

    # Image path existence
    print("\n  Image path check (first 500 train):")
    missing = 0
    for r in train[:500]:
        if not Path(r["img_path"]).exists():
            missing += 1
    print(f"    Missing: {missing}/500 (first 500 train)")

    # Sample weight distribution
    weights = Counter(r["sample_weight"] for r in train)
    print(f"\n  Sample weight distribution in train: {dict(weights)}")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building plate type classifier manifests (6 classes)...")
    print()

    print("1/6 Scanning CCPD2019...")
    scan_ccpd2019()

    print("2/6 Scanning CCPD2020...")
    scan_ccpd2020()

    print("3/6 Scanning CRPD...")
    scan_crpd()

    print("4/6 Scanning special_v2...")
    scan_special_v2()

    print("5/6 Scanning git_plate...")
    scan_git_plate()

    print("6/6 Scanning CBLPRD...")
    scan_cblprd()

    print(f"\nTotal raw records: {len(records)}")
    by_class = Counter(r["label_name"] for r in records)
    print(f"Raw class distribution: {dict(by_class)}")

    print("\nAssigning splits...")
    train, val_c, val_h, val_cs, holdout = assign_splits()

    # Write outputs
    write_class_map()
    write_csv(train, "train.csv")
    write_csv(val_c, "val_clean.csv")
    write_csv(val_h, "val_hard.csv")
    write_csv(val_cs, "val_cross_source.csv")
    write_final_holdout_template()
    write_source_stats(train, val_c, val_h, val_cs, holdout)
    write_build_summary(train, val_c, val_h, val_cs, holdout)

    print_audit(train, val_c, val_h, val_cs, holdout)

    print(f"\nOutput: {OUT_DIR}")

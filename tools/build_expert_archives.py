#!/usr/bin/env python3
"""
build_expert_archives.py — 构建专家模型归档包

根据 expert_archive_plan.json 为四类专家生成 model_zoo 归档结构。
默认 dry-run。支持逐步复制小文件。

用法:
  python tools/build_expert_archives.py                            # dry-run
  python tools/build_expert_archives.py --execute                  # 执行（只复制小文件）
  python tools/build_expert_archives.py --execute --copy-checkpoints   # 包含权重
  python tools/build_expert_archives.py --execute --copy-board-artifacts  # 包含板端产物
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = PROJECT_ROOT / "expert_archive_plan.json"
MODEL_ZOO = PROJECT_ROOT / "model_zoo"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", default=True, dest="dry_run")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--copy-checkpoints", action="store_true", default=False)
    p.add_argument("--copy-board-artifacts", action="store_true", default=False)
    p.add_argument("--copy-small-files", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    is_dry_run = args.dry_run
    if args.execute:
        is_dry_run = False

    if not PLAN_PATH.exists():
        print(f"ERROR: Plan not found: {PLAN_PATH}")
        sys.exit(1)

    with open(PLAN_PATH) as f:
        plan = json.load(f)

    experts = plan.get("experts", {})

    print(f"{'='*60}")
    print(f"  Build Expert Archives")
    print(f"  Mode: {'DRY-RUN' if is_dry_run else 'EXECUTE'}")
    print(f"  Copy checkpoints: {args.copy_checkpoints}")
    print(f"  Copy board artifacts: {args.copy_board_artifacts}")
    print(f"  Copy small files: {args.copy_small_files}")
    print(f"{'='*60}\n")

    all_actions = []
    for etype, data in experts.items():
        print(f"[{etype}]")
        zoo_dir = MODEL_ZOO / etype
        subdirs = ["checkpoints", "board_artifacts", "configs", "manifests", "logs", "keys"]
        for sd in subdirs:
            d = zoo_dir / sd
            action = {"type": "mkdir", "path": str(d.relative_to(PROJECT_ROOT))}
            all_actions.append(action)
            if not is_dry_run:
                d.mkdir(parents=True, exist_ok=True)

        # Keys files
        keys_file = data.get("keys_file", "")
        if keys_file:
            src = PROJECT_ROOT / keys_file
            dst = zoo_dir / "keys" / Path(keys_file).name
            if src.exists():
                all_actions.append({"type": "copy", "src": str(src.relative_to(PROJECT_ROOT)), "dst": str(dst.relative_to(PROJECT_ROOT)), "size": src.stat().st_size})
                if not is_dry_run and args.copy_small_files:
                    dst.parent.mkdir(exist_ok=True)
                    shutil.copy2(src, dst)

        # Rebased manifests (small files)
        for mf_key in ["rebased_manifest", "rebased_test_manifest"]:
            mf = data.get(mf_key, "")
            if mf and not mf.startswith("NEEDS"):
                src = PROJECT_ROOT / mf
                if src.exists():
                    dst = zoo_dir / "manifests" / Path(mf).name
                    all_actions.append({"type": "copy", "src": str(src.relative_to(PROJECT_ROOT)), "dst": str(dst.relative_to(PROJECT_ROOT)), "size": src.stat().st_size})
                    if not is_dry_run and args.copy_small_files:
                        dst.parent.mkdir(exist_ok=True)
                        shutil.copy2(src, dst)

        # Checkpoints (only if --copy-checkpoints)
        for ckpt in data.get("checkpoints", []):
            src = PROJECT_ROOT / ckpt if not ckpt.startswith("/") else Path(ckpt)
            # If it's a directory, list contents
            if src.is_dir():
                for f in src.rglob("*.pth"):
                    if "Final" in f.name or "best" in f.name:
                        dst = zoo_dir / "checkpoints" / f"{etype}_{f.parent.name}_{f.name}"
                        all_actions.append({"type": "copy_checkpoint", "src": str(f.relative_to(PROJECT_ROOT)), "dst": str(dst.relative_to(PROJECT_ROOT)), "size": f.stat().st_size})
                        if not is_dry_run and args.copy_checkpoints:
                            dst.parent.mkdir(exist_ok=True)
                            shutil.copy2(f, dst)
            elif src.exists():
                dst = zoo_dir / "checkpoints" / f"{etype}_{src.name}"
                all_actions.append({"type": "copy_checkpoint", "src": str(src.relative_to(PROJECT_ROOT)), "dst": str(dst.relative_to(PROJECT_ROOT)), "size": src.stat().st_size})
                if not is_dry_run and args.copy_checkpoints:
                    dst.parent.mkdir(exist_ok=True)
                    shutil.copy2(src, dst)

        # Board artifacts (only if --copy-board-artifacts)
        for ba_key in ["rknn_location", "onnx_location"]:
            ba = data.get(ba_key, "")
            if ba:
                src = PROJECT_ROOT / ba
                if src.exists():
                    dst = zoo_dir / "board_artifacts" / src.name
                    all_actions.append({"type": "copy_board_artifact", "src": str(src.relative_to(PROJECT_ROOT)), "dst": str(dst.relative_to(PROJECT_ROOT)), "size": src.stat().st_size})
                    if not is_dry_run and args.copy_board_artifacts:
                        dst.parent.mkdir(exist_ok=True)
                        shutil.copy2(src, dst)

        print(f"  Planned {len([a for a in all_actions if etype in str(a.get('dst',''))])} actions for {etype}")

    # Summary
    copies = [a for a in all_actions if a["type"] == "copy"]
    checkpoints = [a for a in all_actions if a["type"] == "copy_checkpoint"]
    board_arts = [a for a in all_actions if a["type"] == "copy_board_artifact"]
    mkdirs = [a for a in all_actions if a["type"] == "mkdir"]

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Directories to create: {len(mkdirs)}")
    print(f"  Small files to copy:   {len(copies)} (keys, manifests)")
    print(f"  Checkpoints to copy:   {len(checkpoints)} (only if --copy-checkpoints)")
    print(f"  Board artifacts to copy: {len(board_arts)} (only if --copy-board-artifacts)")

    total_size = sum(a.get("size", 0) for a in all_actions)
    print(f"  Total data: {total_size / 1024 / 1024:.1f} MB" if total_size > 0 else "")

    if is_dry_run:
        print(f"\n  Dry-run complete. Run with --execute to copy small files.")
        print(f"  Add --copy-checkpoints to include weight files.")
        print(f"  Add --copy-board-artifacts to include ONNX/RKNN.")
    else:
        print(f"\n  Execute complete. Archive structure at: {MODEL_ZOO}")

    # Save action log
    log_path = PROJECT_ROOT / "docs" / "expert_archive_dry_run.json" if is_dry_run else PROJECT_ROOT / "docs" / "expert_archive_execute_report.json"
    with open(log_path, "w") as f:
        json.dump({"mode": "dry-run" if is_dry_run else "execute", "actions": all_actions}, f, ensure_ascii=False, indent=2)
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()

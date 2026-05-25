#!/usr/bin/env python3
"""
migrate_workspace.py — LPRNet 工作区迁移工具

功能:
  - 根据 migration_plan.json 执行迁移
  - 默认 dry-run，仅打印计划
  - --execute 执行实际移动
  - 自动创建目标目录
  - 同名文件不覆盖，自动添加后缀
  - 记录迁移操作到 docs/migration_record_YYYY-MM-DD.md
  - 生成 rollback_plan.json

安全:
  - 对 requires_manual_review=True 的项目默认跳过
  - 对 confidence 不是 high 的项目默认跳过
  - 不删除任何文件
  - 不移动数据集目录
  - 不原地修改 manifest 文件
  - 所有操作可回滚

用法:
  python tools/migrate_workspace.py              # dry-run
  python tools/migrate_workspace.py --execute     # 执行
  python tools/migrate_workspace.py --include-review  # 包含需人工审查的项目（谨慎）
  python tools/migrate_workspace.py --include-low      # 包含低置信度项目（极谨慎）
"""

import json
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
MIGRATION_PLAN = PROJECT_ROOT / "migration_plan.json"

# 禁止移动的目录模式
PROTECTED_DIR_PATTERNS = [
    "datasets/",
    "data/",
    "src/",
    "scripts/",
    ".git/",
    ".conda/",
    ".hermes/",
]
# 禁止移动的文件模式
PROTECTED_FILE_PATTERNS = [
    "train.txt", "val.txt", "test.txt",
    "train.csv", "val.csv", "test.csv",
]

DRY_RUN = True
INCLUDE_REVIEW = False
INCLUDE_LOW = False


def parse_args():
    global DRY_RUN, INCLUDE_REVIEW, INCLUDE_LOW
    args = sys.argv[1:]
    if "--execute" in args:
        DRY_RUN = False
    if "--include-review" in args:
        INCLUDE_REVIEW = True
    if "--include-low" in args:
        INCLUDE_LOW = True
        INCLUDE_REVIEW = True  # low 需要 review


def is_protected(path: str) -> bool:
    """检查路径是否应受保护"""
    rel_path = path.replace(str(PROJECT_ROOT), "").lstrip("/")
    for pat in PROTECTED_DIR_PATTERNS:
        if rel_path.startswith(pat):
            return True
    for pat in PROTECTED_FILE_PATTERNS:
        if rel_path.endswith(pat):
            return True
    return False


def find_available_path(target_path: Path) -> Path:
    """找到可用的目标路径，避免覆盖"""
    if not target_path.exists():
        return target_path
    counter = 1
    while True:
        new_path = target_path.parent / f"{target_path.name}.bak{counter:03d}"
        if not new_path.exists():
            return new_path
        counter += 1


def execute_migration(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """执行迁移操作，返回 (成功列表, 失败列表)"""
    successful = []
    failed = []
    record_lines = []

    # 创建目标目录
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    record_lines.append(f"# 工作区迁移记录 - {date.today()}")
    record_lines.append(f"\n> 生成时间: {timestamp}")
    record_lines.append(f"> 模式: {'DRY-RUN' if DRY_RUN else '实际执行'}")
    record_lines.append(f"")
    record_lines.append("| 状态 | 原路径 | 新路径 | 类型 | 原因 |")
    record_lines.append("|------|--------|--------|------|------|")

    for idx, item in enumerate(items):
        old_path = item["old_path"]
        new_path = item["new_path"]
        reason = item["reason"]
        file_type = item["file_type"]
        confidence = item["confidence"]
        requires_review = item["requires_manual_review"]

        # 安全检查
        if not new_path:
            # 新路径为空表示应删除，跳过
            failed.append({
                "item": item,
                "error": "new_path 为空（该操作需要删除），跳过"
            })
            continue

        if is_protected(old_path):
            failed.append({
                "item": item,
                "error": "受保护路径，跳过"
            })
            continue

        src = Path(old_path)
        dst = Path(new_path)

        if not src.exists():
            failed.append({
                "item": item,
                "error": f"源路径不存在: {old_path}"
            })
            continue

        if dst.exists():
            # 查找可用路径
            dst = find_available_path(dst)
            reason += f" (目标已存在，重命名为 {dst.name})"

        if DRY_RUN:
            status = "🟡 计划"
            rel_src = str(src.relative_to(PROJECT_ROOT)) if src.is_relative_to(PROJECT_ROOT) else str(src)
            rel_dst = str(dst.relative_to(PROJECT_ROOT)) if dst.is_relative_to(PROJECT_ROOT) else str(dst)
            record_lines.append(f"| {status} | {rel_src} | {rel_dst} | {file_type} | {reason} |")
            successful.append({
                "item": item,
                "planned_action": f"mv {old_path} -> {new_path}",
                "actual_dst": str(dst)
            })
            continue

        # 实际执行
        try:
            # 确保目标目录存在
            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                shutil.move(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))

            status = "✅ 成功"
            rel_src = str(src.relative_to(PROJECT_ROOT)) if src.is_relative_to(PROJECT_ROOT) else str(src)
            rel_dst = str(dst.relative_to(PROJECT_ROOT)) if dst.is_relative_to(PROJECT_ROOT) else str(dst)
            record_lines.append(f"| {status} | {rel_src} | {rel_dst} | {file_type} | {reason} |")

            successful.append({
                "item": item,
                "old_path": old_path,
                "new_path": str(dst),
                "status": "moved"
            })
        except Exception as e:
            status = "❌ 失败"
            record_lines.append(f"| {status} | {old_path} | {new_path} | {file_type} | 错误: {e} |")
            failed.append({
                "item": item,
                "error": str(e)
            })

    # 写入迁移记录
    if DRY_RUN:
        record_path = DOCS_DIR / f"migration_dryrun_{date.today()}.md"
    else:
        record_path = DOCS_DIR / f"migration_record_{date.today()}.md"

    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w", encoding="utf-8") as f:
        f.write("\n".join(record_lines))

    print(f"\n[迁移记录] 已写入: {record_path}")

    return successful, failed


def generate_rollback_plan(successful: list[dict]) -> list[dict]:
    """生成回滚计划"""
    rollback = []
    for s in successful:
        if "old_path" in s and "new_path" in s:
            rollback.append({
                "action": "mv",
                "source": s["new_path"],
                "target": s["old_path"],
                "reason": "回滚迁移"
            })
    return rollback


def main():
    parse_args()

    print(f"{'='*60}")
    print(f" LPRNet 工作区迁移工具")
    print(f"{'='*60}")
    print(f" 模式: {'DRY-RUN (不执行任何移动)' if DRY_RUN else '实际执行'}")
    if INCLUDE_REVIEW:
        print(f" 包含需人工审查的项目: 是")
    if INCLUDE_LOW:
        print(f" 包含低置信度项目: 是")
    print(f"{'='*60}\n")

    # 加载迁移计划
    if not MIGRATION_PLAN.exists():
        print(f"❌ 未找到迁移计划文件: {MIGRATION_PLAN}")
        print(f"   请先运行工具生成 migration_plan.json")
        sys.exit(1)

    with open(MIGRATION_PLAN, encoding="utf-8") as f:
        plan = json.load(f)

    print(f"加载迁移计划: {len(plan)} 条记录")

    # 过滤
    filtered = []
    skipped_review = 0
    skipped_confidence = 0
    skipped_protected = 0

    for item in plan:
        # 跳过需要人工审查的项目
        if item["requires_manual_review"] and not INCLUDE_REVIEW:
            skipped_review += 1
            continue

        # 跳过非高置信度项目
        if item["confidence"] != "high" and not INCLUDE_LOW:
            skipped_confidence += 1
            continue

        # 跳过受保护路径
        if is_protected(item["old_path"]):
            skipped_protected += 1
            continue

        filtered.append(item)

    print(f"  - 需要人工审查: {skipped_review} (跳过)")
    print(f"  - 低置信度: {skipped_confidence} (跳过)")
    print(f"  - 受保护路径: {skipped_protected} (跳过)")
    print(f"  - 准备执行: {len(filtered)}")

    if not filtered:
        print("\n没有可执行的项目。尝试使用 --include-review 或 --include-low")
        return

    # 按类型分类并打印预览
    by_type = {}
    for item in filtered:
        ft = item["file_type"]
        by_type.setdefault(ft, []).append(item)

    print(f"\n{'='*60}")
    print(f" 迁移预览（按类型）")
    print(f"{'='*60}")
    for ftype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{ftype}] {len(items)} 项:")
        for item in items[:5]:
            old = item["old_path"]
            new = item["new_path"]
            if old and new:
                # 只显示相对路径
                rel_old = old.replace(str(PROJECT_ROOT) + "/", "")
                rel_new = new.replace(str(PROJECT_ROOT) + "/", "")
                print(f"    {rel_old}")
                print(f"      → {rel_new}")
        if len(items) > 5:
            print(f"    ... 还有 {len(items) - 5} 项")

    if DRY_RUN:
        print(f"\n{'='*60}")
        print(f" DRY-RUN 模式 — 未执行任何操作")
        print(f" 使用 --execute 参数执行实际迁移")
        print(f" 使用 --include-review 包括需人工审查的项目")
        print(f" 使用 --include-low 包括低置信度项目")
        print(f"{'='*60}")
    else:
        print(f"\n即将执行 {len(filtered)} 项迁移操作...")
        print(f"按 Ctrl+C 取消，或等待 3 秒后自动继续...")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n已取消")
            return

        print(f"\n开始执行迁移...\n")

        successful, failed = execute_migration(filtered)

        # 生成回滚计划
        rollback_plan = generate_rollback_plan(successful)
        rollback_path = PROJECT_ROOT / "rollback_plan.json"
        with open(rollback_path, "w", encoding="utf-8") as f:
            json.dump(rollback_plan, f, ensure_ascii=False, indent=2)
        print(f"[回滚计划] 已写入: {rollback_path}")

        # 结果汇总
        print(f"\n{'='*60}")
        print(f" 迁移结果")
        print(f"{'='*60}")
        print(f" 成功: {len(successful)}")
        print(f" 失败: {len(failed)}")
        if failed:
            print(f"\n 失败明细:")
            for f_item in failed[:10]:
                print(f"   - {f_item['item']['old_path']}: {f_item['error']}")
            if len(failed) > 10:
                print(f"   ... 还有 {len(failed) - 10} 个失败")

    # 安全汇总
    total_skipped = skipped_review + skipped_confidence + skipped_protected
    print(f"\n{'='*60}")
    print(f" 安全总结")
    print(f"{'='*60}")
    print(f" 计划总数: {len(plan)}")
    print(f" 已跳过: {total_skipped}")
    print(f" 已执行: {len(filtered)}")
    if not DRY_RUN:
        print(f" 已成功: {len(successful)}")
        if failed:
            print(f" ⚠️ 有 {len(failed)} 个失败，请检查日志")
    else:
        print(f" ⚡ DRY-RUN — 未执行任何文件操作")


if __name__ == "__main__":
    main()

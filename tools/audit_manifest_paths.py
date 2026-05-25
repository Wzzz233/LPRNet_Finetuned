#!/usr/bin/env python3
"""
audit_manifest_paths.py — 只读扫描 manifest 文件的路径依赖

功能:
  1. 扫描项目中的 manifest 文件 (train.txt, val.txt, *.csv, *.json, *.jsonl, *.lst)
  2. 识别其中的路径字段，判断类型（绝对/相对/裸文件名）
  3. 检查路径是否存在
  4. 统计每份 manifest 的路径健康度
  5. 生成 docs/manifest_path_audit.md 和 manifest_path_inventory.json

运行方式:
  python tools/audit_manifest_paths.py
"""

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

# CLI args (optional — overridden by main() when provided)
MANIFEST_SCAN_ROOT = PROJECT_ROOT  # where to scan for manifests
OUTPUT_MD = "manifest_path_audit.md"  # relative to DOCS_DIR
OUTPUT_JSON = "manifest_path_inventory.json"  # relative to PROJECT_ROOT

# 扩展名白名单
MANIFEST_EXTS = {".txt", ".csv", ".json", ".jsonl", ".lst"}
# 已知 manifest 文件名模式
MANIFEST_PATTERNS = [
    r"train.*\.(txt|csv|jsonl?|lst)",
    r"val.*\.(txt|csv|jsonl?|lst)",
    r"test.*\.(txt|csv|jsonl?|lst)",
    r".*manifest.*\.csv",
    r".*manifest.*\.json",
    r".*\.(train|val|test)\.csv",
    r"proxy_.*\.csv",
    r"extreme.*\.csv",
    r"summary.*\.json",
]

# 最大行采样数（超大 manifest 只采样）
MAX_SAMPLE_LINES = 5000
# 路径检查超时（秒）
PATH_CHECK_TIMEOUT = 0.05

# CSV/JSON 中可能包含路径的字段名
PATH_FIELD_HINTS = [
    "img_path", "image_path", "path", "filepath", "file_path",
    "img_rel_path", "rel_path", "image", "img", "filename",
    "label_path", "label", "gt_path", "annotation_path",
    "crop_path", "ocrin_path", "dump_path",
]


def is_manifest_file(filepath: Path) -> bool:
    """判断文件是否可能是 manifest"""
    name = filepath.name
    if filepath.suffix.lower() in MANIFEST_EXTS:
        # 按名称模式匹配
        for pat in MANIFEST_PATTERNS:
            if re.match(pat, name, re.IGNORECASE):
                return True
        # 大文件但名称含 manifest/split/proxy 关键词
        keywords = ["manifest", "train", "val", "test", "proxy", "extreme",
                     "split", "index", "sample", "list", "batch", "bucket"]
        if any(kw in name.lower() for kw in keywords):
            return True
    return False


def find_manifest_files(scan_root=None):
    '''递归查找所有 manifest 文件'''
    if scan_root is None:
        scan_root = PROJECT_ROOT
    
    manifests = []
    # 扫描 manifests/ 目录（仅在默认扫描根时）
    if scan_root == PROJECT_ROOT and MANIFESTS_DIR.exists():
        for f in sorted(MANIFESTS_DIR.rglob('*')):
            if f.is_file() and is_manifest_file(f):
                manifests.append(f)

    # 扫描 scan_root 下的所有 manifest
    for ext in ['.txt', '.csv', '.lst', '.json', '.jsonl']:
        for f in sorted(scan_root.rglob(f'*{ext}')):
            if f.is_file() and is_manifest_file(f) and f not in manifests:
                manifests.append(f)

    return manifests


def sample_lines(filepath: Path, max_lines: int = MAX_SAMPLE_LINES) -> list[str]:
    """安全采样文件行，对超大文件只取首尾"""
    total_lines = 0
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                total_lines += 1
    except Exception:
        pass

    if total_lines == 0:
        return []

    if total_lines <= max_lines:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return [l.rstrip("\n\r") for l in f]

    # 大文件：取头部、中间、尾部
    lines = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for i, l in enumerate(f):
                if i < max_lines // 3:
                    lines.append(l.rstrip("\n\r"))
                elif i >= total_lines - max_lines // 3:
                    lines.append(l.rstrip("\n\r"))
                elif i == total_lines // 2:
                    # 取中间的一小段
                    mid_lines = []
                    for _ in range(max_lines // 3):
                        try:
                            mid_lines.append(next(f).rstrip("\n\r"))
                        except StopIteration:
                            break
                    lines.extend(mid_lines)
    except Exception:
        pass

    return lines


def classify_path_type(path_str: str) -> str:
    """判断路径类型：absolute / relative / bare / url / empty"""
    path_str = path_str.strip()
    if not path_str or path_str == "nan" or path_str == "None":
        return "empty"
    if path_str.startswith(("http://", "https://", "ftp://")):
        return "url"
    if path_str.startswith("/"):
        return "absolute"
    if path_str.startswith(("./", "../")):
        return "relative"
    if "/" not in path_str and "\\" not in path_str:
        return "bare"
    return "relative"


def extract_paths_from_csv(filepath: Path, sample: list[str]) -> dict:
    """从 CSV manifest 中提取路径"""
    result = {
        "total_rows": 0,
        "paths": [],
        "header": [],
        "path_columns": [],
        "sample_records": [],
    }

    if not sample:
        return result

    try:
        dialect = csv.Sniffer().sniff(sample[0], delimiters=",;\t")
        reader = csv.reader(sample, delimiter=dialect.delimiter)
    except Exception:
        reader = csv.reader(sample)

    rows = list(reader)
    if not rows:
        return result

    result["header"] = rows[0]
    result["total_rows"] = len(rows) - 1  # 减去 header

    # 找路径列
    header_lower = [h.lower().strip() for h in rows[0]]
    path_cols = []
    for i, h in enumerate(header_lower):
        if any(hint in h for hint in PATH_FIELD_HINTS) or h in ("img_path", "path"):
            path_cols.append(i)
    result["path_columns"] = [rows[0][i] for i in path_cols]

    # 如果没有显式路径列，尝试常见的第 0 列或包含斜杠的列
    if not path_cols:
        for col_idx in range(min(3, len(rows[0]))):
            sample_vals = [r[col_idx] for r in rows[1:min(10, len(rows))] if len(r) > col_idx]
            if any("/" in v or v.startswith("/") for v in sample_vals):
                path_cols.append(col_idx)
        result["path_columns"] = [rows[0][i] if i < len(rows[0]) else f"col{i}" for i in path_cols]

    # 提取路径
    seen = set()
    for r in rows[1:]:
        for col_idx in path_cols:
            if col_idx < len(r):
                val = r[col_idx].strip()
                if val and val not in seen and val not in ("nan", "None", ""):
                    seen.add(val)
                    result["paths"].append(val)

    # 采样记录
    for r in rows[1:min(6, len(rows))]:
        result["sample_records"].append(dict(zip(rows[0], r)))

    return result


def extract_paths_from_text(filepath: Path, sample: list[str]) -> dict:
    """从纯文本 manifest（每行一个路径）中提取路径"""
    result = {
        "total_rows": 0,
        "paths": [],
        "sample_records": [],
    }

    for line in sample:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("//"):
            result["paths"].append(line)
    result["total_rows"] = len(result["paths"])

    for line in result["paths"][:5]:
        result["sample_records"].append({"path": line})

    return result


def extract_paths_from_json(filepath: Path, sample: list[str]) -> dict:
    """从 JSON/JSONL manifest 中提取路径"""
    result = {
        "total_rows": 0,
        "paths": [],
        "field_names": set(),
        "sample_records": [],
    }

    for line in sample:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # 递归找路径字段
        def find_paths(obj, depth=0):
            if depth > 3:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k_lower = k.lower()
                    if any(hint in k_lower for hint in PATH_FIELD_HINTS):
                        if isinstance(v, str) and v:
                            result["paths"].append(v)
                            result["field_names"].add(k)
                    elif isinstance(v, (dict, list)):
                        find_paths(v, depth + 1)
            elif isinstance(obj, list) and depth == 0:
                for item in obj[:100]:
                    find_paths(item, depth + 1)

        find_paths(obj)
        if len(result["sample_records"]) < 3:
            result["sample_records"].append(obj)

    result["total_rows"] = len(sample)
    return result


def analyze_manifest(filepath: Path) -> dict:
    """分析单个 manifest 文件"""
    result = {
        "manifest_path": str(filepath.relative_to(PROJECT_ROOT)),
        "absolute_path": str(filepath.resolve()),
        "file_size_bytes": filepath.stat().st_size,
        "file_size_readable": "",
        "total_rows": 0,
        "effective_samples": 0,
        "absolute_paths": 0,
        "relative_paths": 0,
        "bare_filenames": 0,
        "url_paths": 0,
        "empty_paths": 0,
        "valid_paths": 0,
        "invalid_paths": 0,
        "error_paths": [],
        "sample_paths": [],
        "path_distribution": {},
        "manifest_type": "unknown",
        "risks": [],
        "risk_level": "low",
        "dataset_roots": set(),
        "is_large_file": False,
        "parse_error": None,
    }

    sz = filepath.stat().st_size
    if sz < 1024:
        result["file_size_readable"] = f"{sz}B"
    elif sz < 1024 * 1024:
        result["file_size_readable"] = f"{sz/1024:.1f}K"
    else:
        result["file_size_readable"] = f"{sz/1024/1024:.1f}M"

    if sz > 50 * 1024 * 1024:
        result["is_large_file"] = True

    # 采样行
    sample = sample_lines(filepath)
    if not sample:
        # 空文件
        result["effective_samples"] = 0
        return result

    result["total_rows"] = len(sample)

    # 根据扩展名选择解析策略
    ext = filepath.suffix.lower()

    try:
        if ext == ".csv":
            parsed = extract_paths_from_csv(filepath, sample)
            result["manifest_type"] = "csv"
        elif ext == ".json" or ext == ".jsonl":
            parsed = extract_paths_from_json(filepath, sample)
            result["manifest_type"] = "json" if ext == ".json" else "jsonl"
        else:
            parsed = extract_paths_from_text(filepath, sample)
            result["manifest_type"] = "text"
    except Exception as e:
        result["parse_error"] = str(e)
        # fallback 纯文本
        parsed = extract_paths_from_text(filepath, sample)
        result["manifest_type"] = "text_fallback"

    result["effective_samples"] = len(parsed.get("paths", []))

    # 分类路径
    for p in parsed.get("paths", []):
        ptype = classify_path_type(p)
        if ptype == "absolute":
            result["absolute_paths"] += 1
        elif ptype == "relative":
            result["relative_paths"] += 1
        elif ptype == "bare":
            result["bare_filenames"] += 1
        elif ptype == "url":
            result["url_paths"] += 1
        else:
            result["empty_paths"] += 1

        # 检查路径是否存在
        full_path = None
        if p.startswith("/"):
            full_path = Path(p)
        elif p.startswith(("./", "../")):
            full_path = (filepath.parent / p).resolve()
        else:
            full_path = PROJECT_ROOT / p

        exists = full_path.exists() if full_path else False
        if exists:
            result["valid_paths"] += 1
        else:
            result["invalid_paths"] += 1
            if len(result["error_paths"]) < 20:
                result["error_paths"].append({
                    "path": p,
                    "resolved": str(full_path) if full_path else "N/A",
                    "error": "not_found"
                })

    # 统计分布
    result["path_distribution"] = {
        "absolute": result["absolute_paths"],
        "relative": result["relative_paths"],
        "bare": result["bare_filenames"],
        "url": result["url_paths"],
        "empty": result["empty_paths"],
        "valid": result["valid_paths"],
        "invalid": result["invalid_paths"],
    }

    # 采样展示路径
    for p in parsed.get("paths", [])[:10]:
        result["sample_paths"].append(p)

    # 推断 dataset_root
    abs_paths = [p for p in parsed.get("paths", []) if p.startswith("/")]
    if abs_paths:
        # 找共同前缀
        common = os.path.commonprefix(abs_paths)
        common_parts = common.split("/")
        # 取前几级作为 dataset_root
        for depth in range(min(len(common_parts), 6), 2, -1):
            candidate = "/".join(common_parts[:depth])
            result["dataset_roots"].add(candidate)

    # 风险评级
    if result["absolute_paths"] > 0:
        result["risks"].append(f"含 {result['absolute_paths']} 个绝对路径，迁移后失效风险高")
        result["risk_level"] = "high"
    if result["invalid_paths"] > 0:
        result["risks"].append(f"含 {result['invalid_paths']} 个无效路径")
        if result["risk_level"] == "low":
            result["risk_level"] = "medium"
    if result["is_large_file"]:
        result["risks"].append(f"大文件 ({result['file_size_readable']})，采样可能不完整")
    if result["parse_error"]:
        result["risks"].append(f"解析错误: {result['parse_error']}")

    return result


def generate_markdown_report(all_results: list[dict]) -> str:
    """生成 Markdown 审计报告"""
    lines = []
    lines.append("# Manifest 路径依赖审计报告")
    lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n扫描路径: {PROJECT_ROOT}")
    lines.append(f"\n---\n")

    # 统计总览
    total_manifests = len(all_results)
    high_risk = sum(1 for r in all_results if r["risk_level"] == "high")
    med_risk = sum(1 for r in all_results if r["risk_level"] == "medium")
    total_abs = sum(r["absolute_paths"] for r in all_results)
    total_invalid = sum(r["invalid_paths"] for r in all_results)
    total_rows = sum(r["effective_samples"] for r in all_results)

    lines.append("## 总览\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 扫描 manifest 数 | {total_manifests} |")
    lines.append(f"| 高风险 | {high_risk} |")
    lines.append(f"| 中风险 | {med_risk} |")
    lines.append(f"| 低风险 | {total_manifests - high_risk - med_risk} |")
    lines.append(f"| 绝对路径总数 | {total_abs} |")
    lines.append(f"| 无效路径总数 | {total_invalid} |")
    lines.append(f"| 采样样本总行数 | {total_rows} |")
    lines.append("")

    # 风险详情表
    lines.append("## 每份 manifest 详情\n")
    lines.append("| 文件 | 类型 | 大小 | 绝对路径 | 相对路径 | 裸文件名 | 有效 | 无效 | 风险 |")
    lines.append("|------|------|------|---------|---------|---------|------|------|------|")

    for r in sorted(all_results, key=lambda x: x["risk_level"] + x["manifest_path"], reverse=True):
        lines.append(
            f"| {r['manifest_path']} "
            f"| {r['manifest_type']} "
            f"| {r['file_size_readable']} "
            f"| {r['absolute_paths']} "
            f"| {r['relative_paths']} "
            f"| {r['bare_filenames']} "
            f"| {r['valid_paths']} "
            f"| {r['invalid_paths']} "
            f"| {r['risk_level']} |"
        )

    lines.append("")

    # 高风险 manifest 详细分析
    high_risk_manifests = [r for r in all_results if r["risk_level"] == "high"]
    if high_risk_manifests:
        lines.append("## 高风险 Manifest 分析\n")
        for r in high_risk_manifests[:20]:
            lines.append(f"### {r['manifest_path']}\n")
            lines.append(f"- 类型: {r['manifest_type']}")
            lines.append(f"- 大小: {r['file_size_readable']}")
            lines.append(f"- 采样行数: {r['total_rows']} (总计约{max(r['total_rows'], 0)})")
            lines.append(f"- 绝对路径: {r['absolute_paths']}")
            lines.append(f"- 有效路径: {r['valid_paths']}")
            lines.append(f"- 无效路径: {r['invalid_paths']}")

            if r["dataset_roots"]:
                for dr in sorted(r["dataset_roots"]):
                    lines.append(f"- 疑似 dataset_root: `{dr}`")

            if r["error_paths"]:
                lines.append("\n无效路径示例:")
                for ep in r["error_paths"][:5]:
                    lines.append(f"  - `{ep['path']}` → {ep.get('resolved', 'N/A')}")

            if r["sample_paths"]:
                lines.append("\n路径示例:")
                for sp in r["sample_paths"][:5]:
                    lines.append(f"  - `{sp}`")

            if r["risks"]:
                lines.append("\n风险说明:")
                for risk in r["risks"]:
                    lines.append(f"  - ⚠️ {risk}")

            lines.append("")

    # 路径类型分布汇总
    lines.append("## 路径类型分布汇总\n")
    lines.append("| 类型 | 合计 |")
    lines.append("|------|------|")
    lines.append(f"| 绝对路径 | {total_abs} |")
    lines.append(f"| 相对路径 | {sum(r['relative_paths'] for r in all_results)} |")
    lines.append(f"| 裸文件名 | {sum(r['bare_filenames'] for r in all_results)} |")
    lines.append(f"| URL | {sum(r['url_paths'] for r in all_results)} |")
    lines.append(f"| 空值 | {sum(r['empty_paths'] for r in all_results)} |")
    lines.append(f"| 有效 | {sum(r['valid_paths'] for r in all_results)} |")
    lines.append(f"| 无效 | {total_invalid} |")
    lines.append("")

    # 风险与建议
    lines.append("## 风险与建议\n")
    if total_abs > 0:
        lines.append(f"### ⛔ 绝对路径风险 (发现 {total_abs} 个绝对路径)\n")
        lines.append("绝对路径绑定到当前机器路径 `/home/wzzz/`。如果：")
        lines.append("- 迁移到其他机器或容器")
        lines.append("- 变更用户名或路径")
        lines.append("- 在云端环境运行")
        lines.append("则所有绝对路径立即失效。\n")
        lines.append("**建议**: 统一改为 `dataset_root + relative_path` 格式。")
        lines.append("对应的 dataset_root 可在训练配置中声明。\n")

    if total_invalid > 0:
        lines.append(f"### ❌ 无效路径风险 (发现 {total_invalid} 个无效路径)\n")
        lines.append("部分 manifest 中的路径在当前文件系统下不存在。可能原因：")
        lines.append("- 数据集路径已变更")
        lines.append("- 数据被移动或删除")
        lines.append("- 软链接断链\n")
        lines.append("**建议**: 首先检查软链接是否正常，然后验证数据完整性。\n")

    lines.append("### 推荐迁移方案\n")
    lines.append("1. **不改动原始 manifest** — 所有修改通过生成新 manifest 实现")
    lines.append("2. **使用软链接兼容** — 对旧绝对路径通过 `ln -s` 兼容")
    lines.append("3. **dataset_root 配置化** — 训练脚本读取 config 中的 `dataset_root` 字段")
    lines.append("4. **新 manifest 统一相对路径** — 所有路径基于 `dataset_root`\n")

    lines.append("---\n")
    lines.append(f"*报告由 audit_manifest_paths.py 自动生成，{len(all_results)} 份 manifest 已扫描*")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="只读扫描 manifest 文件的路径依赖")
    parser.add_argument("--manifest-root", type=str, default=None,
                        help="指定 manifest 扫描根目录（默认：PROJECT_ROOT）")
    parser.add_argument("--output-md", type=str, default=None,
                        help="输出 Markdown 报告路径")
    parser.add_argument("--output-json", type=str, default=None,
                        help="输出 JSON 清单路径")
    args = parser.parse_args()
    
    os.chdir(PROJECT_ROOT)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    scan_root = Path(args.manifest_root).resolve() if args.manifest_root else PROJECT_ROOT
    output_md = Path(args.output_md).resolve() if args.output_md else DOCS_DIR / "manifest_path_audit.md"
    output_json = Path(args.output_json).resolve() if args.output_json else PROJECT_ROOT / "manifest_path_inventory.json"

    print(f"[audit_manifest_paths.py] 扫描 manifest 文件...")
    print(f"[audit_manifest_paths.py] 项目根目录: {PROJECT_ROOT}")
    print(f"[audit_manifest_paths.py] 扫描根目录: {scan_root}")
    
    manifests = find_manifest_files(scan_root)
    print(f"[audit_manifest_paths.py] 找到 {len(manifests)} 个 manifest 文件")

    # 分析每个 manifest
    all_results = []
    errors = 0
    for i, mf in enumerate(manifests):
        try:
            print(f"  [{i+1}/{len(manifests)}] {mf.relative_to(PROJECT_ROOT)}", end="")
            result = analyze_manifest(mf)
            all_results.append(result)
            risk_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(result["risk_level"], "⚪")
            print(f" → {result['risk_level']} {risk_icon} abs={result['absolute_paths']} inv={result['invalid_paths']}")
        except Exception as e:
            errors += 1
            print(f" → ❌ 错误: {e}")
            all_results.append({
                "manifest_path": str(mf.relative_to(PROJECT_ROOT)),
                "absolute_path": str(mf.resolve()),
                "file_size_bytes": mf.stat().st_size if mf.exists() else 0,
                "file_size_readable": "N/A",
                "total_rows": 0,
                "effective_samples": 0,
                "absolute_paths": 0,
                "relative_paths": 0,
                "bare_filenames": 0,
                "url_paths": 0,
                "empty_paths": 0,
                "valid_paths": 0,
                "invalid_paths": 0,
                "error_paths": [],
                "sample_paths": [],
                "path_distribution": {},
                "manifest_type": "error",
                "risks": [f"解析错误: {e}"],
                "risk_level": "unknown",
                "dataset_roots": set(),
                "is_large_file": False,
                "parse_error": str(e),
            })

    # 为 JSON 序列化转换 set -> list
    serializable_results = []
    for r in all_results:
        r["dataset_roots"] = list(r["dataset_roots"])
        serializable_results.append(r)

    # 写入 JSON 清单
    json_path = output_json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    print(f"\n[audit_manifest_paths.py] 已写入: {json_path}")

    # 写入 Markdown 报告
    report = generate_markdown_report(all_results)
    md_path = output_md
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[audit_manifest_paths.py] 已写入: {md_path}")

    # 汇总统计
    total_abs = sum(r["absolute_paths"] for r in all_results)
    total_invalid = sum(r["invalid_paths"] for r in all_results)
    total_samples = sum(r["effective_samples"] for r in all_results)

    print(f"\n=== 审计摘要 ===")
    print(f"  扫描文件数: {len(manifests)}")
    print(f"  采样样本行: {total_samples}")
    print(f"  绝对路径数: {total_abs}")
    print(f"  无效路径数: {total_invalid}")
    print(f"  高风险文件: {sum(1 for r in all_results if r['risk_level'] == 'high')}")
    print(f"  解析错误: {errors}")
    print(f"  完成状态: {'有错误' if errors > 0 else '全部完成'} ✅")

    return 0


if __name__ == "__main__":
    sys.exit(main())

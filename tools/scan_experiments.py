#!/usr/bin/env python3
"""
scan_experiments.py — 只读扫描训练实验目录

功能:
  1. 自动识别疑似训练实验目录
  2. 查找权重文件 (*.pth, *.pt, *.ckpt, *.onnx, *.engine)
  3. 查找日志文件和配置文件
  4. 尝试从日志中提取训练指标
  5. 生成 docs/experiment_index.md 和 experiments_inventory.json

运行方式:
  python tools/scan_experiments.py
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = PROJECT_ROOT / "runs"

# 权重文件扩展名
WEIGHT_EXTS = {".pth", ".pt", ".ckpt", ".onnx", ".engine", ".rknn", ".weights"}
# 日志文件扩展名
LOG_EXTS = {".log", ".txt"}
# 配置扩展名
CONFIG_EXTS = {".yaml", ".yml", ".json", ".toml", ".sh", ".cfg"}

METRIC_PATTERNS = [
    (r"(?:best|val|test|eval).?(?:acc|accuracy)[:\s]*([0-9.]+)", "accuracy"),
    (r"train[_\s]?(?:loss|acc)[:\s]*([0-9.]+)", "train_metric"),
    (r"(?:val|test|eval)[_\s]?loss[:\s]*([0-9.]+)", "val_loss"),
    (r"train[_\s]?loss[:\s]*([0-9.]+)", "train_loss"),
    (r"(?:epoch|Epoch)[\s#]*(\d+)", "epoch"),
    (r"(?:lr|learning_rate)[:\s]*([0-9.e+\-]+)", "lr"),
    (r"(?:batch[_\s]?size|batch_size)[:\s]*(\d+)", "batch_size"),
    (r"(?:max_epoch|n_epoch)[:\s]*(\d+)", "max_epoch"),
    (r"best[_\s]?(?:epoch|step)[:\s]*(\d+)", "best_epoch"),
    (r"(?:final|test)[_\s]?(?:acc|accuracy)[:\s]*([0-9.]+)", "final_acc"),
    (r"proxy[_\s]?acc[:\s]*([0-9.]+)", "proxy_acc"),
    (r"board[_\s]?acc[:\s]*([0-9.]+)", "board_acc"),
    (r"(?:green|blue)[_\s]?acc[:\s]*([0-9.]+)", "color_acc"),
    (r"mAP[0-9]*[:\s]*([0-9.]+)", "mAP"),
    (r"precision[:\s]*([0-9.]+)", "precision"),
    (r"recall[:\s]*([0-9.]+)", "recall"),
    (r"f1[:\s]*([0-9.]+)", "f1"),
]


def is_experiment_dir(path: Path) -> bool:
    """判断目录是否可能是实验目录"""
    name = path.name.lower()
    # 排除非实验目录
    exclude_keywords = [
        "__pycache__", ".git", ".conda", ".claude", ".hermes", ".telecodex",
        "node_modules", ".ipynb_checkpoints"
    ]
    if any(kw in name for kw in exclude_keywords):
        return False
    # 实验目录通常包含以下特征
    has_weight = any(f.suffix.lower() in WEIGHT_EXTS for f in path.iterdir() if f.is_file())
    has_log = any(f.name == "train.log" for f in path.iterdir() if f.is_file())
    has_summary = any(f.name in ("train_summary.json", "summary.json") for f in path.iterdir() if f.is_file())
    has_eval = any("eval" in f.name.lower() and f.suffix in (".json", ".csv") for f in path.iterdir() if f.is_file())
    has_eval_log = any("eval" in f.name.lower() and f.suffix == ".txt" for f in path.iterdir() if f.is_file())

    return has_weight or has_log or has_summary or has_eval or has_eval_log


def find_experiment_dirs() -> list[Path]:
    """递归查找实验目录"""
    dirs = []

    # 扫描 experiments/
    if EXPERIMENTS_DIR.exists():
        for d in sorted(EXPERIMENTS_DIR.iterdir()):
            if d.is_dir():
                if is_experiment_dir(d):
                    dirs.append(d)
                # 一级子目录也可能是实验
                for sub in sorted(d.iterdir()):
                    if sub.is_dir() and is_experiment_dir(sub) and sub not in dirs:
                        dirs.append(sub)

    # 扫描 runs/ (极有可能包含实验)
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            if d.is_dir():
                if is_experiment_dir(d):
                    dirs.append(d)
                for sub in sorted(d.iterdir()):
                    if sub.is_dir() and is_experiment_dir(sub) and sub not in dirs:
                        dirs.append(sub)

    return dirs


def parse_timestamp_from_name(name: str) -> str | None:
    """从目录名中提取日期时间"""
    # 模式: YYYYMMDD_HHMMSS 或 YYYYMMDD-HHMMSS 或 YYYYMMDD
    patterns = [
        r"(\d{8})[_-]?(\d{6})",
        r"(\d{8})",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            try:
                if len(m.groups()) == 2:
                    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                else:
                    dt = datetime.strptime(m.group(1), "%Y%m%d")
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return None


def find_weight_files(exp_dir: Path) -> list[dict]:
    """查找实验目录中的权重文件"""
    weights = []
    for f in exp_dir.iterdir():
        if f.is_file() and f.suffix.lower() in WEIGHT_EXTS:
            weights.append({
                "path": str(f.relative_to(PROJECT_ROOT)),
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "size_readable": "",
            })
    return weights


def find_log_files(exp_dir: Path) -> list[Path]:
    """查找日志文件"""
    logs = []
    for f in exp_dir.iterdir():
        if f.is_file() and f.suffix.lower() in LOG_EXTS:
            logs.append(f)
        if f.is_file() and f.name in ("nohup.out", "train.log", "eval.log", "test.log", "output.log"):
            if f not in logs:
                logs.append(f)
    return logs


def find_config_files(exp_dir: Path) -> list[Path]:
    """查找配置文件"""
    configs = []
    for f in exp_dir.iterdir():
        if f.is_file() and f.suffix.lower() in CONFIG_EXTS:
            configs.append(f)
    return configs


def find_eval_files(exp_dir: Path) -> list[Path]:
    """查找评估结果文件"""
    evals = []
    for f in exp_dir.iterdir():
        if f.is_file() and "eval" in f.name.lower() and f.suffix in (".json", ".csv", ".txt"):
            evals.append(f)
    return evals


def parse_metrics_from_log(log_path: Path) -> dict:
    """从日志中提取指标"""
    metrics = {}
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
        # 取最后 500 行
        lines = content.splitlines()
        tail = "\n".join(lines[-500:])

        for pattern, key in METRIC_PATTERNS:
            matches = re.findall(pattern, tail, re.IGNORECASE)
            if matches:
                values = [float(m) if m.replace(".", "").isdigit() else m for m in matches]
                metrics[key] = values[-1]  # 取最后一个值
    except Exception:
        pass
    return metrics


def parse_metrics_from_summary(exp_dir: Path) -> dict:
    """从 train_summary.json 中提取指标"""
    summary_files = [
        exp_dir / "train_summary.json",
        exp_dir / "summary.json",
    ]
    for sf in summary_files:
        if sf.exists():
            try:
                data = json.loads(sf.read_text(encoding="utf-8", errors="replace"))
                metrics = {}
                for key in ["best_epoch", "best_proxy_acc", "final_test_acc",
                            "best_board_metric", "best_selection_metric",
                            "selection_decode_mode", "selection_strategy",
                            "args"]:
                    if key in data:
                        metrics[key] = data[key]
                return metrics
            except Exception:
                pass
    return {}


def extract_dataset_from_args(metrics: dict) -> str:
    """从训练参数中提取数据集信息"""
    args = metrics.get("args", {})
    if isinstance(args, dict):
        manifest = args.get("train_manifest", args.get("manifest", args.get("txt_file", "")))
        if manifest:
            return manifest
        img_dirs = args.get("train_img_dirs", "")
        if img_dirs:
            return img_dirs
    return "未知"


def scan_experiment(exp_dir: Path) -> dict:
    """扫描单个实验目录"""
    result = {
        "experiment_path": str(exp_dir.relative_to(PROJECT_ROOT)),
        "absolute_path": str(exp_dir.resolve()),
        "directory_name": exp_dir.name,
        "suspected_date": parse_timestamp_from_name(exp_dir.name),
        "size_bytes": sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file()),
        "size_readable": "",
        "weights": find_weight_files(exp_dir),
        "logs": [str(f.relative_to(PROJECT_ROOT)) for f in find_log_files(exp_dir)],
        "configs": [str(f.relative_to(PROJECT_ROOT)) for f in find_config_files(exp_dir)],
        "eval_files": [str(f.relative_to(PROJECT_ROOT)) for f in find_eval_files(exp_dir)],
        "summary_metrics": {},
        "log_metrics": {},
        "dataset": "未知",
        "best_metric": None,
        "best_weight": None,
        "risk_notes": [],
        "recommend_keep": True,
    }

    # 大小格式化
    sz = result["size_bytes"]
    if sz < 1024:
        result["size_readable"] = f"{sz}B"
    elif sz < 1024 * 1024:
        result["size_readable"] = f"{sz/1024:.1f}K"
    else:
        result["size_readable"] = f"{sz/1024/1024:.1f}M"

    # 解析日志
    for log_path in find_log_files(exp_dir):
        metrics = parse_metrics_from_log(log_path)
        result["log_metrics"].update(metrics)

    # 解析 summary
    summary = parse_metrics_from_summary(exp_dir)
    result["summary_metrics"] = summary
    result["dataset"] = extract_dataset_from_args(summary)

    # 找最佳权重
    best_weight = None
    final_weight = None
    for w in result["weights"]:
        name = w["name"]
        if "best" in name.lower():
            best_weight = w
        if "final" in name.lower():
            final_weight = w

    if best_weight:
        result["best_weight"] = best_weight["path"]
    elif final_weight:
        result["best_weight"] = final_weight["path"]
    elif result["weights"]:
        result["best_weight"] = result["weights"][-1]["path"]

    # 最佳指标
    if summary:
        best_acc = summary.get("best_proxy_acc") or summary.get("final_test_acc") or summary.get("best_board_metric")
        if best_acc:
            result["best_metric"] = best_acc

    if not result["best_metric"] and result["log_metrics"]:
        for key in ["accuracy", "final_acc", "proxy_acc", "board_acc"]:
            if key in result["log_metrics"]:
                result["best_metric"] = result["log_metrics"][key]
                break
            elif result["log_metrics"].get(key):
                result["best_metric"] = result["log_metrics"][key]
                break

    # 风险分析
    if not result["weights"]:
        result["risk_notes"].append("无权重文件，可能是中断的实验")
        result["recommend_keep"] = False
    elif len(result["weights"]) < 2:
        result["risk_notes"].append("权重文件过少，可能未完成训练")
        result["recommend_keep"] = False

    if not result["logs"]:
        result["risk_notes"].append("无日志文件，可能已废弃")
        result["recommend_keep"] = False

    if result["best_metric"] is not None and result["best_metric"] < 0.1:
        result["risk_notes"].append(f"最佳指标异常低 ({result['best_metric']:.4f})")

    if not result["configs"]:
        result["risk_notes"].append("无配置文件，可能重复不可复现")

    # 判断是否值得保留
    has_best_or_final = any(
        "best" in w["name"].lower() or "final" in w["name"].lower()
        for w in result["weights"]
    )
    has_log = len(result["logs"]) > 0
    has_summary = bool(summary)
    result["recommend_keep"] = has_best_or_final and (has_log or has_summary)

    return result


def format_best_metric(metric) -> str:
    """格式化最佳指标"""
    if metric is None:
        return "-"
    try:
        return f"{float(metric):.4f}"
    except (ValueError, TypeError):
        return str(metric)


def generate_markdown_report(results: list[dict]) -> str:
    """生成 Markdown 实验索引报告"""
    lines = []
    lines.append("# 实验索引报告")
    lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"扫描路径: {EXPERIMENTS_DIR}")
    lines.append(f"\n---\n")

    # 汇总
    total = len(results)
    with_weights = sum(1 for r in results if r["weights"])
    with_logs = sum(1 for r in results if r["logs"])
    with_best = sum(1 for r in results if r["best_metric"] is not None)
    recommend_keep = sum(1 for r in results if r["recommend_keep"])
    print(f"  total={total}, weights={with_weights}, logs={with_logs}, best={with_best}, keep={recommend_keep}")

    lines.append("## 总览\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 实验目录总数 | {total} |")
    lines.append(f"| 含权重的实验 | {with_weights} |")
    lines.append(f"| 含日志的实验 | {with_logs} |")
    lines.append(f"| 有最佳指标 | {with_best} |")
    lines.append(f"| 建议保留 | {recommend_keep} |")
    lines.append(f"| 建议归档 | {total - recommend_keep} |")
    lines.append("")

    # 总大小
    total_size = sum(r["size_bytes"] for r in results)
    if total_size > 1024**3:
        size_str = f"{total_size/1024**3:.1f}G"
    elif total_size > 1024**2:
        size_str = f"{total_size/1024**2:.1f}M"
    else:
        size_str = f"{total_size/1024:.1f}K"
    lines.append(f"实验总大小: {size_str}\n")

    # 实验明细表
    lines.append("## 实验明细\n")
    lines.append("| # | 实验目录 | 日期 | 大小 | 权重数 | 日志 | 最佳指标 | 数据集/Manifest | 建议 |")
    lines.append("|---|---------|------|------|--------|------|---------|----------------|------|")

    for i, r in enumerate(sorted(results, key=lambda x: x["size_bytes"], reverse=True), 1):
        date = r["suspected_date"] or "未知"
        short_name = r["directory_name"][:60]
        if len(r["directory_name"]) > 60:
            short_name = r["directory_name"][:57] + "..."

        weight_count = len(r["weights"])
        log_count = len(r["logs"])
        best = format_best_metric(r["best_metric"])
        dataset = r["dataset"][:40] if len(r["dataset"]) > 40 else r["dataset"]
        keep = "✅" if r["recommend_keep"] else "⚠️"

        lines.append(
            f"| {i} | {short_name} | {date} | {r['size_readable']} "
            f"| {weight_count} | {log_count} | {best} | {dataset} | {keep} |"
        )

    lines.append("")

    # 建议保留的实验
    keep_exps = [r for r in results if r["recommend_keep"]]
    if keep_exps:
        lines.append("## 建议保留的实验\n")
        lines.append("这些实验包含权重、日志和指标，是重要的实验记录。\n")
        for r in keep_exps[:30]:
            lines.append(f"- {r['directory_name']}")
            lines.append(f"  - 大小: {r['size_readable']}, 最佳指标: {format_best_metric(r['best_metric'])}")
            if r["best_weight"]:
                lines.append(f"  - 最佳权重: `{r['best_weight']}`")
            lines.append("")
        if len(keep_exps) > 30:
            lines.append(f"... 还有 {len(keep_exps) - 30} 个实验\n")

    # 建议归档的实验
    archive_exps = [r for r in results if not r["recommend_keep"]]
    if archive_exps:
        lines.append("## 建议归档/清理的实验\n")
        lines.append("这些实验缺少关键文件（权重/日志），可能是中途失败、未完成或已废弃的。\n")
        for r in archive_exps[:30]:
            notes = "; ".join(r["risk_notes"]) if r["risk_notes"] else "缺少关键文件"
            lines.append(f"- {r['directory_name']} ({r['size_readable']})")
            lines.append(f"  - {notes}")
            lines.append("")
        if len(archive_exps) > 30:
            lines.append(f"... 还有 {len(archive_exps) - 30} 个实验\n")

    # 按大小排序的 top 实验
    sorted_by_size = sorted(results, key=lambda x: x["size_bytes"], reverse=True)[:20]
    lines.append("## 最大实验 (Top 20)\n")
    lines.append("| # | 实验 | 大小 | 权重数 | 最佳指标 | 建议 |")
    lines.append("|---|------|------|--------|---------|------|")
    for i, r in enumerate(sorted_by_size, 1):
        lines.append(
            f"| {i} | {r['directory_name'][:50]} | {r['size_readable']} "
            f"| {len(r['weights'])} | {format_best_metric(r['best_metric'])} "
            f"| {'保留' if r['recommend_keep'] else '归档'} |"
        )
    lines.append("")

    lines.append("---\n")
    lines.append("*报告由 scan_experiments.py 自动生成*")

    return "\n".join(lines)


def main():
    os.chdir(PROJECT_ROOT)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[scan_experiments.py] 扫描实验目录...")
    print(f"[scan_experiments.py] 实验目录: {EXPERIMENTS_DIR}")

    dirs = find_experiment_dirs()
    print(f"[scan_experiments.py] 发现 {len(dirs)} 个疑似实验目录")

    # 并发扫描
    results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(scan_experiment, d): d for d in dirs}
        for i, future in enumerate(as_completed(future_map), 1):
            d = future_map[future]
            try:
                result = future.result()
                results.append(result)
                wc = len(result["weights"])
                lc = len(result["logs"])
                keep = "✅" if result["recommend_keep"] else "⚠️"
                print(f"  [{i}/{len(dirs)}] {result['directory_name'][:60]} | {wc}w {lc}l | {result['size_readable']} | {keep}")
            except Exception as e:
                errors += 1
                print(f"  [{i}/{len(dirs)}] ❌ {d.name}: {e}")
                results.append({
                    "experiment_path": str(d.relative_to(PROJECT_ROOT)),
                    "absolute_path": str(d.resolve()),
                    "directory_name": d.name,
                    "suspected_date": None,
                    "size_bytes": 0,
                    "size_readable": "0",
                    "weights": [],
                    "logs": [],
                    "configs": [],
                    "eval_files": [],
                    "summary_metrics": {},
                    "log_metrics": {},
                    "dataset": "未知",
                    "best_metric": None,
                    "best_weight": None,
                    "risk_notes": [f"扫描错误: {e}"],
                    "recommend_keep": False,
                })

    # 写入 JSON
    json_path = DOCS_DIR / "experiments_inventory.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[scan_experiments.py] 已写入: {json_path}")

    # 生成 Markdown 报告
    report = generate_markdown_report(results)
    md_path = DOCS_DIR / "experiment_index.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[scan_experiments.py] 已写入: {md_path}")

    # 汇总
    total_size = sum(r["size_bytes"] for r in results)
    total_size_gb = total_size / 1024**3
    keep_count = sum(1 for r in results if r["recommend_keep"])
    archive_count = len(results) - keep_count

    print(f"\n=== 扫描摘要 ===")
    print(f"  实验目录数: {len(dirs)}")
    print(f"  有效扫描: {len(results)}")
    print(f"  总大小: {total_size_gb:.1f}G")
    print(f"  建议保留: {keep_count}")
    print(f"  建议归档: {archive_count}")
    print(f"  扫描错误: {errors}")
    print(f"  完成状态: {'有错误' if errors > 0 else '全部完成'} ✅")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
rebase_manifest_paths.py — Manifest 绝对路径转相对路径转换工具

根据 manifest_rebase_plan.json 中的规则，将 manifest 中的
绝对路径 (/home/wzzz/LPRNet/...) 转换为相对路径。

安全:
  - 默认 dry-run（采样分析，不全量转换）
  - --execute 才生成 manifests_rebased/
  - 不修改任何旧 manifest
  - 只处理 can_auto_rebase=true 的 manifest

用法:
  python tools/rebase_manifest_paths.py --plan manifest_rebase_plan.json --dry-run
  python tools/rebase_manifest_paths.py --plan manifest_rebase_plan.json --execute
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
ABSOLUTE_PATH = re.compile(r"^/home/wzzz/LPRNet/(.+)$")

DRY_RUN_SAMPLE_SIZE = 10  # For dry-run, only sample this many data rows

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--output-dir", default="manifests_rebased")
    p.add_argument("--dry-run", action="store_true", default=True, dest="dry_run")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def convert_abs_path(value: str) -> tuple[str, bool, str]:
    """Convert absolute path to relative. Returns (new_value, changed, warning_or_empty)"""
    m = ABSOLUTE_PATH.match(value)
    if not m:
        return value, False, ""
    rel = m.group(1)
    full = PROJECT_ROOT / rel
    warning = ""
    if not full.exists() and Path(value).suffix.lower() in (".jpg", ".jpeg", ".png", ".ppm", ".bmp"):
        warning = f"  WARNING: path not found: {rel}"
    return rel, True, warning


def sample_csv(old_path: Path, plan_entry: dict, sample_n: int = DRY_RUN_SAMPLE_SIZE) -> dict:
    """Sample a CSV manifest for dry-run analysis"""
    result = {
        "old_manifest": str(old_path.relative_to(PROJECT_ROOT)),
        "new_manifest": plan_entry["new_manifest"],
        "detected_root": plan_entry["detected_root"],
        "status": "sample",
        "converted": 0,
        "total_sampled": 0,
        "warnings": [],
        "sample_before": [],
        "sample_after": [],
        "needs_full_scan": False,
    }
    
    try:
        total_lines = 0
        abs_lines = 0
        rel_lines = 0
        with open(old_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = None
            for i, row in enumerate(reader):
                if header is None:
                    header = row
                    result["header"] = header
                    continue
                total_lines += 1
                
                # Check each field for absolute paths
                for cell in row:
                    if ABSOLUTE_PATH.match(cell):
                        abs_lines += 1
                        break
                else:
                    rel_lines += 1
                
                # Sample a few rows
                if len(result["sample_before"]) < sample_n:
                    converted_row = list(row)
                    for j, cell in enumerate(row):
                        new_val, changed, warn = convert_abs_path(cell)
                        if changed:
                            converted_row[j] = new_val
                            result["converted"] += 1
                        if warn:
                            result["warnings"].append(f"  row {i}: {warn}")
                    result["sample_before"].append(dict(zip(header, row)))
                    result["sample_after"].append(dict(zip(header, converted_row)))
        
        result["total_sampled"] = total_lines if total_lines <= sample_n else sample_n
        result["total_lines"] = total_lines
        result["abs_lines"] = abs_lines
        result["rel_lines"] = rel_lines
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def sample_text(old_path: Path, plan_entry: dict, sample_n: int = DRY_RUN_SAMPLE_SIZE) -> dict:
    """Sample a text manifest for dry-run analysis"""
    result = {
        "old_manifest": str(old_path.relative_to(PROJECT_ROOT)),
        "new_manifest": plan_entry["new_manifest"],
        "detected_root": plan_entry["detected_root"],
        "status": "sample",
        "converted": 0,
        "total_sampled": 0,
        "warnings": [],
        "sample_before": [],
        "sample_after": [],
    }
    
    try:
        total = 0
        with open(old_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total += 1
                if len(result["sample_before"]) < sample_n:
                    line = line.rstrip("\n\r")
                    if not line or line.startswith("#"):
                        continue
                    new_val, changed, warn = convert_abs_path(line)
                    if changed:
                        result["converted"] += 1
                    if warn:
                        result["warnings"].append(warn)
                    result["sample_before"].append(line)
                    result["sample_after"].append(new_val)
        result["total_sampled"] = min(total, sample_n)
        result["total_lines"] = total
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def render_dryrun_report(results: list[dict], validation_items: list[dict], auto_count: int) -> str:
    """Generate dry-run markdown report"""
    lines = []
    lines.append("# Manifest Rebase Dry-Run Report")
    lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模式: DRY-RUN（仅采样分析，未生成任何新 manifest）\n")

    successes = sum(1 for r in results if r["status"] == "sample")
    errors = sum(1 for r in results if r["status"] == "error")
    total_warnings = sum(len(r.get("warnings", [])) for r in results)
    
    lines.append("## 总览\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| Auto-rebase manifest | {auto_count} |")
    lines.append(f"| 采样成功 | {successes} |")
    lines.append(f"| 采样失败 | {errors} |")
    lines.append(f"| Warning 总数 | {total_warnings} |")
    lines.append(f"| 需验证 (validation_required) | {len(validation_items)} |")
    lines.append("")
    
    if validation_items:
        lines.append("## 需验证的 Manifest\n")
        lines.append("以下 manifest 旧根目录软链接缺失，但 canonical 路径存在，rebase 后应可修复：\n")
        for v in validation_items:
            lines.append(f"- [{v['old_manifest']}]({v['old_manifest']})")
            lines.append(f"  - detected_root: {v['detected_root']}")
            lines.append(f"  - broken_before: {v['broken_entries_before_rebase']}, broken_after: {v['broken_entries_after_rebase']}")
            lines.append(f"  - {v['notes']}")
            lines.append("")
    
    # Group by directory
    by_dir = {}
    for r in results:
        d = Path(r["old_manifest"]).parent
        by_dir.setdefault(str(d), []).append(r)
    
    for d, items in sorted(by_dir.items()):
        lines.append(f"## {d}/\n")
        for r in items:
            notes = r.get("notes", "")
            val_flag = " [VALIDATE]" if notes and "需验证" in notes else ""
            abs_line = f"abs={r.get('abs_lines', '?')} rel={r.get('rel_lines', '?')}" if r.get('abs_lines') is not None else ""
            lines.append(f"### {Path(r['old_manifest']).name}{val_flag}\n")
            lines.append(f"- total_lines={r.get('total_lines', '?')}, {abs_line}")
            lines.append(f"- converted_paths_in_sample={r.get('converted', 0)}")
            
            for w in r.get("warnings", []):
                lines.append(f"- ⚠️ {w}")
            
            # Show sample conversions
            for bi, (bef, aft) in enumerate(zip(r.get("sample_before", []), r.get("sample_after", []))):
                if isinstance(bef, dict):
                    # CSV sample - show first path column
                    bef_str = next((v for v in bef.values() if "/" in str(v)), str(bef))
                    aft_str = next((v for v in aft.values() if "/" in str(v)), str(aft))
                else:
                    bef_str = bef
                    aft_str = aft
                
                if str(bef_str) != str(aft_str):
                    lines.append(f"  Before[{bi}]: {str(bef_str)[:80]}")
                    lines.append(f"  After [{bi}]: {str(aft_str)[:80]}")
            lines.append("")
    
    lines.append("---\n")
    lines.append("*DRY-RUN 完成，未执行任何写入操作*")
    return "\n".join(lines)


def main():
    args = parse_args()
    is_dry_run = args.dry_run
    if args.execute:
        is_dry_run = False
    
    # Load plan
    with open(Path(args.plan)) as f:
        plan = json.load(f)
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    
    print(f"{'='*65}")
    print(f"  Manifest Path Rebase Tool")
    print(f"  Mode: {'DRY-RUN (sample only)' if is_dry_run else 'EXECUTE'}")
    print(f"  Plan: {args.plan}")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}\n")
    
    auto_manifests = [p for p in plan if p["can_auto_rebase"]]
    validation_items = [p for p in auto_manifests if p["validation_required"]]
    review_manifests = [p for p in plan if not p["can_auto_rebase"] and p["absolute_path_count"] > 0]
    
    print(f"  Auto-rebase: {len(auto_manifests)} ({len(validation_items)} need validation)")
    print(f"  Manual review (skipped): {len(review_manifests)}")
    print(f"  No action: {len(plan) - len(auto_manifests) - len(review_manifests)}\n")
    
    if is_dry_run:
        # Dry-run: sample each manifest for analysis only
        results = []
        for i, entry in enumerate(auto_manifests):
            old_path = PROJECT_ROOT / entry["old_manifest"]
            val_flag = " [VALIDATE]" if entry["validation_required"] else ""
            print(f"  [{i+1}/{len(auto_manifests)}] {entry['old_manifest'][:60]}{val_flag}", end="")
            
            if not old_path.exists():
                print(f" ⚠ SKIP (not found)")
                continue
            
            ext = old_path.suffix.lower()
            if ext == ".csv":
                r = sample_csv(old_path, entry)
            else:
                r = sample_text(old_path, entry)
            
            r["notes"] = entry["notes"]
            results.append(r)
            
            c = r.get("converted", 0)
            w = len(r.get("warnings", []))
            print(f" ✓ converted={c} warnings={w}")
        
        # Report
        success = sum(1 for r in results if r["status"] == "sample")
        errors = sum(1 for r in results if r["status"] == "error")
        warnings = sum(len(r.get("warnings", [])) for r in results)
        
        print(f"\n{'='*65}")
        print(f"  DRY-RUN COMPLETE")
        print(f"{'='*65}")
        print(f"  Sampled: {success}")
        print(f"  Errors: {errors}")
        print(f"  Warnings: {warnings}")
        print(f"  Validation required: {len(validation_items)}")
        
        # Generate reports
        report_md = render_dryrun_report(results, validation_items, len(auto_manifests))
        report_path = DOCS_DIR / "manifest_rebase_dry_run.md"
        with open(report_path, "w") as f:
            f.write(report_md)
        print(f"  Report: {report_path}")
        
        json_path = PROJECT_ROOT / "manifest_rebase_dry_run.json"
        # Save minimal JSON (no full samples)
        json_report = {
            "mode": "dry-run",
            "total_auto": len(auto_manifests),
            "validation_required": len(validation_items),
            "sampled": success,
            "errors": errors,
            "warnings": warnings,
            "results": [{k: v for k, v in r.items() if k not in ("sample_before", "sample_after")} for r in results]
        }
        with open(json_path, "w") as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        print(f"  JSON: {json_path}")
        
        # Print validation items
        if validation_items:
            print(f"\n{'='*65}")
            print(f"  VALIDATION REQUIRED")
            print(f"{'='*65}")
            for v in validation_items:
                print(f"  - {v['old_manifest']}")
                print(f"    detected_root={v['detected_root']}")
                print(f"    {v['notes']}")
    
    else:
        # EXECUTE mode: generate rebased manifests
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        
        for i, entry in enumerate(auto_manifests):
            old_path = PROJECT_ROOT / entry["old_manifest"]
            new_rel = entry["new_manifest"]
            new_path = output_dir / new_rel
            val_flag = " [VALIDATE]" if entry["validation_required"] else ""
            
            print(f"  [{i+1}/{len(auto_manifests)}] {entry['old_manifest'][:60]}{val_flag}", end="")
            
            if not old_path.exists():
                print(f" ⚠ SKIP (source not found)")
                continue
            
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                ext = old_path.suffix.lower()
                converted = 0
                warnings = []
                
                if ext == ".csv":
                    with open(old_path, "r", encoding="utf-8", errors="replace") as fin:
                        reader = csv.reader(fin)
                        with open(new_path, "w", encoding="utf-8", newline="") as fout:
                            writer = csv.writer(fout)
                            header = None
                            for row in reader:
                                if header is None:
                                    header = row
                                    writer.writerow(row)
                                    continue
                                new_row = list(row)
                                for j, cell in enumerate(row):
                                    nv, chg, warn = convert_abs_path(cell)
                                    if chg:
                                        new_row[j] = nv
                                        converted += 1
                                    if warn:
                                        warnings.append(warn)
                                writer.writerow(new_row)
                else:
                    # Plain text
                    with open(old_path, "r", encoding="utf-8", errors="replace") as fin:
                        with open(new_path, "w", encoding="utf-8") as fout:
                            for line in fin:
                                line_s = line.rstrip("\n\r")
                                if not line_s or line_s.startswith("#"):
                                    fout.write(line)
                                    continue
                                nv, chg, warn = convert_abs_path(line_s)
                                if chg:
                                    fout.write(nv + "\n")
                                    converted += 1
                                else:
                                    fout.write(line)
                                if warn:
                                    warnings.append(warn)
                
                status = "success"
                print(f" ✓ {converted} conversions")
                if warnings:
                    for w in warnings[:3]:
                        print(f"    {w}")
                
                results.append({
                    "old_manifest": entry["old_manifest"],
                    "new_manifest": str(new_path.relative_to(PROJECT_ROOT)),
                    "status": status,
                    "converted": converted,
                    "warnings_count": len(warnings),
                })
                
            except Exception as e:
                print(f" ✗ ERROR: {e}")
                results.append({
                    "old_manifest": entry["old_manifest"],
                    "new_manifest": str(new_path.relative_to(PROJECT_ROOT)),
                    "status": "error",
                    "error": str(e),
                })
        
        # Report
        success = sum(1 for r in results if r["status"] == "success")
        errors = sum(1 for r in results if r["status"] == "error")
        
        print(f"\n{'='*65}")
        print(f"  EXECUTION COMPLETE")
        print(f"{'='*65}")
        print(f"  Successful: {success}")
        print(f"  Errors: {errors}")
        print(f"  Output: {output_dir}")
        print(f"  Mode: Actual conversion (old manifests untouched)")
        
        # Generate execute report
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        md = [f"# Manifest Rebase Execute Report", f"\n> 生成时间: {ts}", f"\n## 总览\n", f"| 指标 | 值 |", f"|------|-----|", f"| 处理成功 | {success} |", f"| 处理失败 | {errors} |", f"| 输出目录 | {output_dir} |", f"| Validation required | {len(validation_items)} |\n"]
        md.append("## 处理明细\n| # | Manifest | 状态 | 转换数 | Warnings |")
        md.append("|---|----------|------|--------|----------|")
        for i, r in enumerate(results, 1):
            md.append(f"| {i} | {r.get('old_manifest','')[:55]} | {'✅' if r['status']=='success' else '❌'} | {r.get('converted',0)} | {r.get('warnings_count',0)} |")
        
        with open(DOCS_DIR / "manifest_rebase_execute_report.md", "w") as f:
            f.write("\n".join(md))
        print(f"  Report: {DOCS_DIR / 'manifest_rebase_execute_report.md'}")
        
        with open(PROJECT_ROOT / "manifest_rebase_execute_report.json", "w") as f:
            json.dump({"timestamp": ts, "total": len(auto_manifests), "successful": success, "errors": errors, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"  JSON: {PROJECT_ROOT / 'manifest_rebase_execute_report.json'}")

    print(f"\n  Run --execute to generate rebased manifests")
    print(f"  (⚠ Not executed — dry-run did not create any files)")


if __name__ == "__main__":
    main()

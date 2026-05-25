#!/usr/bin/env python3
"""tidy_workspace_root.py — 根目录外围文件整理工具"""
import argparse, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    is_dry = args.dry_run if not args.execute else False
    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"Plan not found: {plan_path}")
        sys.exit(1)
    with open(plan_path) as f:
        plan = json.load(f)
    print(f"{'='*55}")
    print(f"  Tidy Workspace Root")
    print(f"  Mode: {'DRY-RUN' if is_dry else 'EXECUTE'}")
    print(f"{'='*55}\n")
    movable = [x for x in plan if x.get("can_move_now") and not x.get("would_overwrite")]
    blocked = [x for x in plan if not x.get("can_move_now")]
    skipped_overwrite = [x for x in plan if x.get("can_move_now") and x.get("would_overwrite")]
    results = []
    for item in movable:
        src = Path(item["old_path"])
        dst = Path(item["new_path"])
        entry = {**item, "status": "ok"}
        if is_dry:
            print(f"  Would move: {src.name} -> {dst.parent.name}/")
            results.append(entry)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                entry["status"] = "skipped_exists"
                print(f"  SKIP (exists): {src.name}")
            else:
                shutil.move(str(src), str(dst))
                entry["status"] = "moved"
                print(f"  MOVED: {src.name} -> {dst.parent.name}/")
            results.append(entry)
    print(f"\n  Plan items: {len(plan)}")
    print(f"  Movable: {len(movable)}")
    print(f"  Blocked (core assets): {len(blocked)}")
    print(f"  Skipped (would overwrite): {len(skipped_overwrite)}")
    if is_dry:
        print(f"\n  Dry-run complete. Run with --execute to move files.")
    else:
        print(f"\n  Execute complete.")
    log_name = "root_tidy_dry_run.json" if is_dry else "root_tidy_execute_report.json"
    log_path = ROOT / log_name
    with open(log_path, "w") as f:
        json.dump({"mode": "dry-run" if is_dry else "execute", "items": results}, f, ensure_ascii=False, indent=2)
    print(f"  Log: {log_path}")
    # Rollback plan
    if not is_dry:
        rollback = [{"action": "mv", "source": x["new_path"], "target": x["old_path"]} for x in results if x["status"] == "moved"]
        with open(ROOT / "root_tidy_rollback_plan.json", "w") as f:
            json.dump(rollback, f, ensure_ascii=False, indent=2)
        print(f"  Rollback: root_tidy_rollback_plan.json")

if __name__ == "__main__":
    main()

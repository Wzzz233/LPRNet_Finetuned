import json
from pathlib import Path
ROOT = Path("/home/wzzz/LPRNet")
MODEL_ZOO = ROOT / "model_zoo"

checks = {}
for etype in sorted(MODEL_ZOO.iterdir()):
    if not etype.is_dir():
        continue
    d = MODEL_ZOO / etype.name
    info = {
        "expert_type": etype.name,
        "README_exists": (d / "README.md").exists(),
        "lineage_exists": (d / "lineage.json").exists(),
        "keys_exists": len(list(d.rglob("*.txt"))) > 0,
        "manifest_exists": len(list(d.rglob("*.csv"))) > 0,
        "checkpoint_exists": len(list(d.rglob("*.pth"))) > 0,
        "board_artifacts_exist": len(list(d.rglob("*.rknn"))) + len(list(d.rglob("*.onnx"))) > 0,
        "keys_list": [str(p.relative_to(ROOT)) for p in d.rglob("*.txt")],
        "manifest_list": [str(p.relative_to(ROOT)) for p in d.rglob("*.csv")],
        "checkpoint_list": [str(p.relative_to(ROOT)) for p in d.rglob("*.pth")],
        "board_artifact_list": [str(p.relative_to(ROOT)) for p in d.rglob("*.rknn")] + [str(p.relative_to(ROOT)) for p in d.rglob("*.onnx")],
    }
    info["missing_items"] = [k for k, v in [("README", info["README_exists"]), ("lineage.json", info["lineage_exists"]), ("keys", info["keys_exists"]), ("manifest", info["manifest_exists"]), ("checkpoint", info["checkpoint_exists"]), ("board_artifact", info["board_artifacts_exist"])] if not v]
    if not info["missing_items"]:
        info["status"] = "archive_ready"
    elif len(info["missing_items"]) <= 2:
        info["status"] = "archive_partial"
    else:
        info["status"] = "needs_manual_identification"
    checks[etype.name] = info

for etype, info in sorted(checks.items()):
    icon = {"archive_ready": "✅", "archive_partial": "🟡", "needs_manual_identification": "❌"}.get(info["status"], "?")
    print(f"{icon} {etype:30s} missing={info['missing_items']}")

with open(ROOT / "model_zoo_completeness_review.json", "w") as f:
    json.dump(checks, f, ensure_ascii=False, indent=2)
print(f"\nWritten: model_zoo_completeness_review.json")

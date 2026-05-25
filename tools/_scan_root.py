import json
from pathlib import Path
ROOT = Path("/home/wzzz/LPRNet")
items = []
for f in sorted(ROOT.iterdir()):
    if f.is_dir() and not f.name.startswith("."):
        items.append({"path": f.name, "type": "dir"})
    elif f.is_file():
        items.append({"path": f.name, "type": "file", "size": f.stat().st_size})
CORE_DIRS = {"datasets","data","experiments","manifests","manifests_rebased","model_zoo","src","configs","keys","scripts","tools","docs","reports","labels","generated","tmp","archive","artifacts","runs","weights_tiny_province_h20b","green_edgefit_v4_boardlike_a3000","third_party","tests","qa_firstchar_hires_rawcrop","qa_firstchar_hires_rawcrop_abc","qa_firstchar_patch_ab","qa_firstchar_patch_ab_v2","qa_u1c_green_generation_difficulty"}
results = []
movable = 0
for item in items:
    p = item["path"]
    entry = {"path": p, "file_type": item["type"], "is_core_asset": False, "can_move_now": False, "risk": "low", "reason": "", "recommendation": "keep"}
    if p in CORE_DIRS:
        entry["is_core_asset"] = True; entry["risk"] = "high"; entry["reason"] = "核心目录"; entry["recommendation"] = "blocked"
    elif item["type"] == "file" and p.endswith(".md"):
        if p in ("README.md","AI使用披露.md","Gemini报告.md","LPRNet_YOLO_车牌识别系统技术报告.md","true_quad_refiner_robust_plan.md"):
            entry["reason"] = "正式文档"
        else:
            entry["can_move_now"] = True; entry["reason"] = "报告"; entry["recommendation"] = "move_to_docs"; movable += 1
    elif item["type"] == "file" and p.endswith(".py"):
        if p in ("augment_image.py","generate_chars_image.py","generate_plate_template.py"):
            entry["reason"] = "symlink 脚本"
        else:
            entry["can_move_now"] = True; entry["reason"] = "临时脚本"; entry["recommendation"] = "move_to_tools"; movable += 1
    elif item["type"] == "file" and p.endswith(".json"):
        entry["can_move_now"] = True; entry["reason"] = "中间 JSON"; entry["recommendation"] = "move_to_generated_json"; movable += 1
    elif item["type"] == "file" and p.endswith(".onnx"):
        entry["is_core_asset"] = True; entry["risk"] = "high"; entry["reason"] = "板端 ONNX"; entry["recommendation"] = "blocked"
    elif item["type"] == "file" and "Zone.Identifier" in p:
        entry["can_move_now"] = True; entry["reason"] = "Windows 残留"; entry["recommendation"] = "move_to_cleanup"; movable += 1
    elif item["type"] == "file" and p.startswith("hermes_"):
        entry["can_move_now"] = True; entry["reason"] = "对话缓存"; entry["recommendation"] = "move_to_cleanup"; movable += 1
    results.append(entry)
for r in results:
    flag = "MOVE" if r["can_move_now"] else ("BLOCK" if r["recommendation"] == "blocked" else "KEEP")
    print(f"  [{flag:5s}] {r['path']}")
print(f"\nTotal: {len(results)}, Movable: {movable}")
with open("root_cleanup_review.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Written: root_cleanup_review.json")

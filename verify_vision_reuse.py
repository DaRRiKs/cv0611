import ast
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["HF_HOME"] = str(ROOT / "artifacts" / "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
nb = json.loads((ROOT / "VLM_практикум_Colab.ipynb").read_text(encoding="utf-8"))
namespace = {"__name__": "vision_reuse_verification"}
for index in [4, 6, 8]:
    exec("".join(nb["cells"][index]["source"]), namespace)
exec("from PIL import Image\nimport re", namespace)
for index in [14, 17]:
    tree = ast.parse("".join(nb["cells"][index]["source"]))
    body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or
            isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "FIELD_QUESTIONS" for target in node.targets)]
    exec(compile(ast.Module(body=body, type_ignores=[]), "definitions", "exec"), namespace)
image, gold = namespace["make_invoice"](seed=7, font_size=14)
rows = []
for edge in [512, 1024]:
    print("Checking resolution", edge, flush=True)
    original, first_info = namespace["extract_by_fields"](image, longest_edge=edge, return_info=True, reuse_vision=False)
    optimized, second_info = namespace["extract_by_fields"](image, longest_edge=edge, return_info=True, reuse_vision=True)
    assert original == optimized, (edge, original, optimized)
    assert sum(item["vision_reused"] for item in second_info.values()) == 3
    rows.append({"longest_edge": edge, "seed": 7, "original": original, "optimized": optimized,
                 "identical_answers": original == optimized,
                 "original_seconds": sum(item["seconds"] for item in first_info.values()),
                 "optimized_seconds": sum(item["seconds"] for item in second_info.values())})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
(ROOT / "artifacts" / "vision_reuse_verification.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

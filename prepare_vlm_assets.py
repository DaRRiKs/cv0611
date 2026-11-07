import json
import os
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(ARTIFACTS / "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import HfApi, snapshot_download


models = [
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    "florence-community/Florence-2-base",
]
revisions = {}
for name in models:
    info = HfApi().model_info(name)
    revisions[name] = info.sha
    print(name, info.sha, flush=True)
(ARTIFACTS / "model_revisions.json").write_text(
    json.dumps(revisions, indent=2), encoding="utf-8"
)

image_path = ARTIFACTS / "cats_original.jpg"
if not image_path.exists():
    errors = []
    for url in [
        "https://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000039769.jpg",
    ]:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                image_path.write_bytes(response.read())
            break
        except Exception as error:
            errors.append(str(error))
    if not image_path.exists():
        print("Image download failed:", errors, flush=True)

for name, revision in revisions.items():
    path = snapshot_download(
        name,
        revision=revision,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
        ignore_patterns=["onnx/*"],
        max_workers=4,
    )
    print("Downloaded:", name, path, flush=True)

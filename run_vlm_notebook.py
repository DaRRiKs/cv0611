from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
os.chdir(ROOT)
os.environ["HF_HOME"] = str(ARTIFACTS / "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["IPYTHONDIR"] = str(ARTIFACTS / "ipython")
os.environ["MPLCONFIGDIR"] = str(ARTIFACTS / "matplotlib")
os.environ["JUPYTER_DATA_DIR"] = str(ARTIFACTS / "jupyter")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ARTIFACTS / "jupyter_runtime")
os.environ["JUPYTER_PATH"] = str(ARTIFACTS / "jupyter")
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
kernel_path = ARTIFACTS / "jupyter" / "kernels" / "vlm"
kernel_path.mkdir(parents=True, exist_ok=True)
kernel = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
          "display_name": "VLM local validation", "language": "python"}
(kernel_path / "kernel.json").write_text(json.dumps(kernel), encoding="utf-8")

import nbformat
from nbclient import NotebookClient
from update_vlm_conclusions import update_conclusions


class LoggingNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg.get("msg_type") == "stream":
            text = msg.get("content", {}).get("text", "")
            if text.startswith("Задание"):
                print(text, end="", flush=True)
        return super().process_message(msg, cell, cell_index)


def save_checkpoint(notebook, execution_status):
    temporary = path.with_suffix(".tmp.ipynb")
    nbformat.write(notebook, temporary)
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)
    status_path = ARTIFACTS / "execution_status.json"
    status_temp = status_path.with_suffix(".tmp")
    status_temp.write_text(json.dumps(execution_status, ensure_ascii=False, indent=2), encoding="utf-8")
    status_temp.replace(status_path)


path = ROOT / "VLM_практикум_Colab.ipynb"
nb = nbformat.read(path, as_version=4)
client = LoggingNotebookClient(nb, timeout=None, startup_timeout=180, kernel_name="vlm",
                        resources={"metadata": {"path": str(ROOT)}}, record_timing=True)
status = {"started_utc": datetime.now(timezone.utc).isoformat(), "completed_cell_ids": [],
          "full_execution": False, "failure": None, "cell_seconds": {}}
started = time.perf_counter()
try:
    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            waiting_for = {22: "HuggingFaceTB/SmolVLM-500M-Instruct", 29: "florence-community/Florence-2-base"}.get(index)
            if waiting_for:
                revision = json.loads((ARTIFACTS / "model_revisions.json").read_text(encoding="utf-8"))[waiting_for]
                weights = ARTIFACTS / "hf_cache" / "hub" / ("models--" + waiting_for.replace("/", "--")) / "snapshots" / revision / "model.safetensors"
                waiting_started = time.perf_counter()
                while not weights.exists():
                    if time.perf_counter() - waiting_started > 1800:
                        raise TimeoutError(f"Weights not downloaded: {waiting_for}")
                    print(f"Waiting for model download: {waiting_for}", flush=True)
                    time.sleep(30)
            print(f"START {index} {cell.id}", flush=True)
            status["active_cell_id"] = cell.id
            save_checkpoint(nb, status)
            cell_started = time.perf_counter()
            client.execute_cell(cell, index, execution_count=len(status["completed_cell_ids"]) + 1)
            status["completed_cell_ids"].append(cell.id)
            status["cell_seconds"][cell.id] = time.perf_counter() - cell_started
            update_conclusions(nb, status["completed_cell_ids"])
            status["active_cell_id"] = None
            save_checkpoint(nb, status)
            print(f"DONE {index} {cell.id}: {status['cell_seconds'][cell.id]:.1f}s", flush=True)
    status["full_execution"] = True
except (Exception, KeyboardInterrupt) as error:
    status["failure"] = {"type": type(error).__name__, "message": str(error)}
    traceback.print_exc()
finally:
    status["elapsed_seconds"] = time.perf_counter() - started
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    update_conclusions(nb, status["completed_cell_ids"])
    save_checkpoint(nb, status)
    print(json.dumps({key: value for key, value in status.items() if key != "failure"}, ensure_ascii=False), flush=True)
if status["failure"]:
    sys.exit(1)

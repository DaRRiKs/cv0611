import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import tokenize

import matplotlib
import nbformat
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "vlm_results"
PATH = ROOT / "VLM_практикум_Colab.ipynb"
sys.stdout.reconfigure(encoding="utf-8")
nb = nbformat.read(PATH, as_version=4)
nbformat.validate(nb)
original = nbformat.read(ROOT / "artifacts" / "original_notebook.ipynb", as_version=4)
assert [cell.id for cell in nb.cells[:len(original.cells)]] == [cell.id for cell in original.cells]
assert nb.cells[37].source == original.cells[37].source
status = json.loads((ROOT / "artifacts" / "execution_status.json").read_text(encoding="utf-8"))
assert status["full_execution"] and status["failure"] is None
code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
assert all(output.output_type != "error" for cell in code_cells for output in cell.outputs)
assert {cell.id for cell in code_cells} == set(status["completed_cell_ids"])
for cell in code_cells:
    tree = ast.parse(cell.source)
    compile(tree, cell.id, "exec")
    assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(cell.source).readline))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None
        assert not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str))

namespace = {"random": random, "Image": Image, "ImageDraw": ImageDraw, "ImageFont": ImageFont,
             "matplotlib": matplotlib, "Path": Path, "hashlib": hashlib, "re": re, "np": np}
for cell in code_cells:
    tree = ast.parse(cell.source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"levenshtein", "anls1", "normalize", "postprocess", "normalize_yes_no",
                                   "make_invoice", "font", "image_digest", "find_knee"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), cell.id, "exec"), namespace)

def verify_field_scores(frame):
    expected_anls = [namespace["anls1"](row.pred, row.gold) for row in frame.itertuples()]
    assert np.allclose(frame.anls.to_numpy(), expected_anls, rtol=0, atol=1e-14)
    assert frame.exact.eq(frame.pred.astype(str).eq(frame.gold.astype(str))).all()
    for row in frame.itertuples():
        field = getattr(row, "field", "invoice_number")
        assert namespace["postprocess"](field, row.raw_answer) == str(row.pred)

task1 = pd.read_csv(RESULTS / "task1_answers_long.csv", keep_default_na=False)
task1_metrics = pd.read_csv(RESULTS / "task1_metrics.csv").set_index("model")
assert len(task1) == 20
assert task1.normalized.tolist() == [namespace["normalize_yes_no"](answer) for answer in task1.raw_answer]
for name, group in task1.groupby("model"):
    assert len(group) == 10 and int(group.gold.eq("yes").sum()) == 5
    assert np.isclose(group.normalized.eq(group.gold).mean(), task1_metrics.loc[name, "accuracy"])
    assert np.isclose(group.loc[group.gold.eq("no"), "normalized"].eq("yes").mean(),
                      task1_metrics.loc[name, "false_positive_rate"])
    assert task1_metrics.loc[name, "ambiguous"] == group.normalized.eq("ambiguous").sum()
    assert task1_metrics.loc[name, "false_yes"] == (group.gold.eq("no") & group.normalized.eq("yes")).sum()

task2 = pd.read_csv(RESULTS / "task2_fields.csv", keep_default_na=False)
task3 = pd.read_csv(RESULTS / "task3_fields.csv", keep_default_na=False)
fields = {"invoice_number", "date", "seller", "total"}
for frame, edges, size in [(task2, [1024, 2048], 160), (task3, [512, 768, 1024, 1536, 2048], 400)]:
    verify_field_scores(frame)
    assert len(frame) == size
    assert set(frame.longest_edge) == set(edges)
    assert not frame.duplicated(["seed", "longest_edge", "field"]).any()
    for edge, group in frame.groupby("longest_edge"):
        assert set(group.seed) == set(range(20)) and set(group.field) == fields
        assert group.groupby("seed").size().eq(4).all()
    assert frame.exact.eq(frame.pred.astype(str).eq(frame.gold.astype(str))).all()
    assert frame.anls.between(0, 1).all()
    assert frame.seconds.gt(0).all()
    assert (frame.visual_tokens.mod(64) == 0).all()
    assert frame.groupby(["seed", "longest_edge"]).vision_reused.sum().eq(3).all()

key = ["seed", "longest_edge", "field"]
comparison = task2.merge(task3, on=key, suffixes=("_2", "_3"), validate="one_to_one")
for column in ["raw_answer", "pred", "exact", "anls", "visual_tokens", "seconds"]:
    assert comparison[f"{column}_2"].equals(comparison[f"{column}_3"])
for number, frame in [(2, task2), (3, task3)]:
    summary = pd.read_csv(RESULTS / f"task{number}_summary.csv").set_index("longest_edge")
    for edge, group in frame.groupby("longest_edge"):
        assert np.isclose(summary.loc[edge, "exact_accuracy"], group.exact.mean())
        assert np.isclose(summary.loc[edge, "anls"], group.anls.mean())
        assert summary.loc[edge, "documents"] == 20
        for field, field_group in group.groupby("field"):
            assert np.isclose(summary.loc[edge, f"exact_{field}"], field_group.exact.mean())

knee = pd.read_csv(RESULTS / "task3_knee.csv").iloc[0]
expected_knee = namespace["find_knee"](pd.read_csv(RESULTS / "task3_summary.csv"))
assert bool(knee.detected) == expected_knee["detected"]
if expected_knee["detected"]:
    assert knee.visual_tokens == expected_knee["visual_tokens"]
    assert str(knee.resolution) == expected_knee["resolution"]

task4 = pd.read_csv(RESULTS / "task4_fields.csv", keep_default_na=False)
task4_summary = pd.read_csv(RESULTS / "task4_summary.csv", dtype={"longest_edge": str})
assert len(task4) == 320
for variant, group in task4.groupby("variant"):
    merged = group.merge(task2[key + ["exact"]], on=key, validate="one_to_one")
    assert merged.actual_error.eq(~merged.exact).all()
    row = task4_summary.loc[task4_summary.variant.eq(variant) & task4_summary.unit.eq("field") &
                            task4_summary.longest_edge.eq("all")].iloc[0]
    for label, mask in {
        "TP": group.actual_error & group.flagged,
        "FN": group.actual_error & ~group.flagged,
        "FP": ~group.actual_error & group.flagged,
        "TN": ~group.actual_error & ~group.flagged,
    }.items():
        assert row[label] == mask.sum()
        assert group.category.eq(label).equals(mask)
    if row.actual_errors:
        assert np.isclose(row.error_detection_rate, row.TP / row.actual_errors)
before = task4.loc[task4.variant.eq("original")].set_index(key)
after = task4.loc[task4.variant.eq("extended")].set_index(key)
assert (after.flagged | ~before.flagged).all()

task5 = pd.read_csv(RESULTS / "task5_fields.csv", keep_default_na=False)
task5_summary = pd.read_csv(RESULTS / "task5_summary.csv").set_index("approach")
task5_delta = pd.read_csv(RESULTS / "task5_delta.csv").iloc[0]
ocr = pd.read_csv(RESULTS / "task5_ocr.csv", keep_default_na=False)
assert len(task5) == 40 and len(ocr) == 20
verify_field_scores(task5)
assert set(ocr.seed) == set(range(20))
assert ocr[["width", "height"]].max(axis=1).eq(512).all()
for approach, group in task5.groupby("approach"):
    assert set(group.seed) == set(range(20))
    assert group.longest_edge.eq(512).all()
    assert np.isclose(group.exact.mean(), task5_summary.loc[approach, "exact_accuracy"])
    assert np.isclose(group.anls.mean(), task5_summary.loc[approach, "anls"])
assert np.isclose(task5_delta.exact_gain_pp,
                  100 * (task5_summary.loc["Hybrid", "exact_accuracy"] - task5_summary.loc["Baseline", "exact_accuracy"]))
assert np.isclose(task5_delta.exact_gain_pp, 100 * (task5_delta.fixed - task5_delta.regressed) / 20)
assert task5_delta[["fixed", "regressed", "both_correct", "both_wrong"]].sum() == 20

records = [json.loads(path.read_text(encoding="utf-8")) for path in (RESULTS / "cache").glob("*.json")]
records = [record for record in records if record["request"]["protocol"] == "vlm-practicum-3-pinned-cache"]
environment = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
question_records = {}
document_records = {}
ocr_records = {}
for record in records:
    request, result = record["request"], record["result"]
    parameters = request["request"]
    assert parameters["do_sample"] is False
    assert request["dtype"] == "torch.float32"
    assert request["revision"] == environment["revisions"][request["model"]]
    if "questions" in parameters:
        document_records[(request["image_sha256"], parameters["edge"])] = result
    elif "question" in parameters:
        question_records[(request["model"], request["image_sha256"], parameters["question"])] = result
    elif parameters.get("task") == "<OCR>":
        ocr_records[request["image_sha256"]] = result

manifest = pd.read_csv(RESULTS / "documents_manifest.csv", keep_default_na=False).set_index("seed")
low_digests = {}
for seed in range(20):
    document, truth = namespace["make_invoice"](seed=seed, font_size=14)
    assert namespace["image_digest"](document) == manifest.loc[seed, "image_sha256"]
    assert all(str(manifest.loc[seed, field]) == str(value) for field, value in truth.items())
    document.thumbnail((512, 512), Image.Resampling.LANCZOS)
    low_digests[seed] = namespace["image_digest"](document)
for row in task3.itertuples():
    assert str(manifest.loc[row.seed, row.field]) == str(row.gold)
    result = document_records[(manifest.loc[row.seed, "image_sha256"], row.longest_edge)]
    assert result["prediction"][row.field] == row.raw_answer
    for key in ["visual_tokens", "prompt_tokens", "vision_reused"]:
        assert result["details"][row.field][key] == getattr(row, key)
    assert np.isclose(result["details"][row.field]["seconds"], row.seconds)

cats_digest = namespace["image_digest"](Image.open(ROOT / "artifacts" / "cats_original.jpg").convert("RGB"))
for row in task1.itertuples():
    model_name = f"HuggingFaceTB/SmolVLM-{row.model}-Instruct"
    assert question_records[(model_name, cats_digest, row.question)]["answer"] == row.raw_answer
ocr_by_seed = ocr.set_index("seed")
for row in task5.itertuples():
    prompt = "What is the invoice number? Answer with the number only."
    if row.approach == "Hybrid":
        prompt = f"Text on the page: {ocr_by_seed.loc[row.seed, 'ocr_text']}\n\n{prompt}"
    key = ("HuggingFaceTB/SmolVLM-256M-Instruct", low_digests[row.seed], prompt)
    assert question_records[key]["answer"] == row.raw_answer
for row in ocr.itertuples():
    assert ocr_records[low_digests[row.seed]]["text"] == row.ocr_text

graph_cell = next(cell for cell in nb.cells if cell.id == "assignment-3-code")
images = [output.data["image/png"] for output in graph_cell.outputs
          if output.output_type in {"display_data", "execute_result"} and "image/png" in output.data]
assert images and all(len(base64.b64decode(image)) > 1000 for image in images)
assert (RESULTS / "task3_resolution_curve.png").stat().st_size > 1000
for number in range(1, 6):
    cell = next(cell for cell in nb.cells if cell.id == f"assignment-{number}-code")
    assert any(output.output_type in {"display_data", "execute_result"} and "text/html" in output.data for output in cell.outputs)
    conclusion = next(cell.source for cell in nb.cells if cell.id == f"assignment-{number}-conclusion")
    assert "пока не выполнен" not in conclusion and "пока не измерена" not in conclusion

report = {"full_notebook_execution": True, "executed_code_cells": len(code_cells),
          "task1_answers": 20, "task2_fields": 160, "task3_fields": 400,
          "task4_validated_predictions_per_validator": 160, "task5_answers": 40,
          "task5_ocr_documents": 20, "metrics_recomputed_from_predictions": True,
          "individual_anls_and_postprocessing_verified": True,
          "prediction_cache_provenance_verified": True,
          "original_cell_order_and_assignment_text_preserved": True,
          "embedded_tables_and_graph_verified": True, "comments": 0, "docstrings": 0,
          "device": json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))["device"]}
(ROOT / "artifacts" / "results_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

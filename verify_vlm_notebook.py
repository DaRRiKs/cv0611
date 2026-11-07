import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tokenize

import nbformat


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "VLM_практикум_Colab.ipynb"
sys.stdout.reconfigure(encoding="utf-8")
nb = nbformat.read(PATH, as_version=4)
nbformat.validate(nb)
namespace = {"__name__": "notebook_validation"}
code_count = 0
for index, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    tree = ast.parse(cell.source, filename=f"cell_{index}")
    compile(tree, f"cell_{index}", "exec")
    code_count += 1
    assert not any(token.type == tokenize.COMMENT for token in
                   tokenize.generate_tokens(io.StringIO(cell.source).readline)), index
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None, (index, getattr(node, "name", "module"))
        assert not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)), (index, "standalone string")
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
              target.id in {"FIELD_QUESTIONS", "ISSUE_FIELDS", "HALLUCINATION_ITEMS"} for target in node.targets):
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, f"cell_{index}_definitions", "exec"), namespace)

namespace["FIELDS"] = list(namespace["FIELD_QUESTIONS"])
ask_source = next(cell.source for cell in nb.cells if cell.cell_type == "code" and "def ask(" in cell.source)
assert "do_sample=False" in ask_source
assert "synchronize(mdl)" in ask_source
assert len(namespace["HALLUCINATION_ITEMS"]) == 10
assert sum(item[1] == "yes" for item in namespace["HALLUCINATION_ITEMS"]) == 5

normalize = namespace["normalize_yes_no"]
for value, expected in {
    "Yes.": "yes", "NO!": "no", "Yes, there is a cat.": "yes",
    "No, there is no dog.": "no", "No, there is not a dog.": "no",
    "Yes or no": "ambiguous", "Maybe yes": "ambiguous", "There is a cat": "yes",
    "There is no dog present in the image.": "no", "There are no cups in the image.": "no",
    "There isn't a dog in the picture.": "no", "The image does not contain a dog.": "no",
    "There might be no dog.": "ambiguous", "I think yes": "ambiguous", "Yes?": "ambiguous",
    "There is no dog, but there is a dog outside.": "ambiguous", "A dog?": "ambiguous",
    "Yes, but I am not sure.": "ambiguous", "No, there is a dog.": "ambiguous",
    "Yes, there is no cat.": "ambiguous", "": "ambiguous",
}.items():
    assert normalize(value) == expected, (value, normalize(value), expected)

make_invoice = namespace["make_invoice"]
image, gold = make_invoice(seed=7, font_size=14)
assert gold == {"invoice_number": "INV-76510", "date": "2026-04-02",
                "seller": "Alpha Supplies LLP", "total": 3615}
for seed in range(20):
    first, first_gold = make_invoice(seed=seed, font_size=14)
    second, second_gold = make_invoice(seed=seed, font_size=14)
    assert first_gold == second_gold
    assert namespace["image_digest"](first) == namespace["image_digest"](second)
    assert first.size == (1240, 1754)
    namespace["date"].fromisoformat(first_gold["date"])
    assert all(row["exact"] for row in namespace["check"](first_gold, first_gold, post=True))

assert namespace["anls1"]("INV12345", "INV-12345") == 8 / 9
assert namespace["anls1"]("abcd", "wxyz") == 0
assert namespace["anls1"]("ABCD", "abcd") == 1
assert namespace["postprocess"]("invoice_number", "The number is INV-12345.") == "INV-12345"
assert namespace["postprocess"]("date", "No readable date") == "No readable date"
assert namespace["validate"](gold) == ["ок"]
assert namespace["validate_extended"](gold) == ["ок"]
impossible_date = {**gold, "date": "2026-02-30"}
assert namespace["validate"](impossible_date) == ["ок"]
assert "несуществующая календарная дата" in namespace["validate_extended"](impossible_date)
assert "несуществующая календарная дата" not in namespace["validate_extended"]({**gold, "date": "2024-02-29"})
assert "несуществующая календарная дата" in namespace["validate_extended"]({**gold, "date": "2026-02-29"})
assert "продавец не указан" in namespace["validate_extended"]({**gold, "seller": ""})

pd = namespace["pd"]
fixture = pd.DataFrame({"actual_error": [True, True, False, False], "flagged": [True, False, True, False]})
metrics = namespace["validation_metrics"](fixture)
assert [metrics[name] for name in ["TP", "FN", "FP", "TN"]] == [1, 1, 1, 1]
assert metrics["error_detection_rate"] == 0.5
assert metrics["precision"] == 0.5
assert pd.isna(namespace["validation_metrics"](fixture.iloc[2:])["error_detection_rate"])
assert not namespace["find_knee"](pd.DataFrame({"visual_tokens": [64, 192, 320],
                                                "exact_accuracy": [0.5, 0.5, 0.5]}))["detected"]
assert not namespace["find_knee"](pd.DataFrame({"visual_tokens": [64, 192, 320],
                                                "exact_accuracy": [0.5, 0.4, 0.8]}))["detected"]
plateau = pd.DataFrame({"visual_tokens": [64, 320, 640, 832],
                       "exact_accuracy": [0.125, 0.80, 0.9875, 0.9875],
                       "longest_edge": [512, 1024, 1536, 2048]})
assert namespace["find_knee"](plateau)["visual_tokens"] == 640

ids = {cell.id for cell in nb.cells}
assert len(ids) == len(nb.cells)
for number in range(1, 6):
    assert f"assignment-{number}-code" in ids
    assert f"assignment-{number}-conclusion" in ids

report = {"nbformat_valid": True, "python_cells_compiled": code_count,
          "comments": 0, "docstrings": 0, "standalone_explanatory_strings": 0,
          "assignments": 5, "helper_checks_passed": True,
          "generated_document_seeds": list(range(20)),
          "fixtures_used_only_for_helper_checks": True,
          "neural_experiments_executed_by_this_script": False}
(ROOT / "artifacts" / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))

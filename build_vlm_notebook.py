import ast
import io
import json
from pathlib import Path
import textwrap
import tokenize


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "VLM_практикум_Colab.ipynb"
BACKUP = ROOT / "artifacts" / "original_notebook.ipynb"
BACKUP.parent.mkdir(exist_ok=True)
if not BACKUP.exists():
    BACKUP.write_bytes(PATH.read_bytes())
nb = json.loads(BACKUP.read_text(encoding="utf-8"))


def clean(source):
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    for index in range(node.body[0].lineno - 1, node.body[0].end_lineno):
                        lines[index] = "\n"
    source = "".join(lines)
    tokens = [token for token in tokenize.generate_tokens(io.StringIO(source).readline)
              if token.type != tokenize.COMMENT]
    return "\n".join(line.rstrip() for line in tokenize.untokenize(tokens).splitlines()).strip() + "\n"


def replace(index, source):
    nb["cells"][index]["source"] = textwrap.dedent(source).strip().splitlines(keepends=True)


def add(kind, source, ident):
    cell = {"cell_type": kind, "id": ident, "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    nb["cells"].append(cell)


replace(3, '''
import importlib.util
import subprocess
import sys

packages = ["transformers==5.22.1", "accelerate==1.15.0", "num2words==0.5.14",
            "Pillow>=10.4,<14", "pandas>=2.2,<4", "matplotlib>=3.9,<4", "requests>=2.32,<3"]
if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
    packages += ["torch>=2.6", "torchvision"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
''')

replace(4, '''
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from IPython.display import display

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(6, os.cpu_count() or 1))
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)

RESULTS_DIR = Path("vlm_results")
RESULTS_DIR.mkdir(exist_ok=True)
REVISIONS = {
    "HuggingFaceTB/SmolVLM-256M-Instruct": "7e3e67edbbed1bf9888184d9df282b700a323964",
    "HuggingFaceTB/SmolVLM-500M-Instruct": "main",
    "florence-community/Florence-2-base": "main",
}
VERSIONS = {name: importlib.metadata.version(name) for name in
            ["torch", "torchvision", "transformers", "accelerate", "Pillow", "numpy", "pandas", "matplotlib", "num2words"]}
ENVIRONMENT = {"python": platform.python_version(), "platform": platform.platform(),
               "device": DEVICE, "dtype": str(DTYPE), "seed": SEED, "versions": VERSIONS,
               "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
               "cpu_threads": torch.get_num_threads(), "revisions": REVISIONS}
(RESULTS_DIR / "environment.json").write_text(json.dumps(ENVIRONMENT, ensure_ascii=False, indent=2), encoding="utf-8")

def synchronize(mdl):
    if mdl.device.type == "cuda":
        torch.cuda.synchronize(mdl.device)

def activate_model(mdl):
    for name in ["model", "model_500", "flor"]:
        other = globals().get(name)
        if other is not None and other is not mdl:
            other.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mdl.to(DEVICE).eval()

print("Устройство:", DEVICE, "|", ENVIRONMENT["gpu"] or "CPU")
print(VERSIONS)
''')

replace(6, '''
from transformers import AutoModelForImageTextToText, AutoProcessor

SMOL = "HuggingFaceTB/SmolVLM-256M-Instruct"
SMOL_500 = "HuggingFaceTB/SmolVLM-500M-Instruct"
FLOR = "florence-community/Florence-2-base"
processor = AutoProcessor.from_pretrained(SMOL, revision=REVISIONS[SMOL])
model = AutoModelForImageTextToText.from_pretrained(SMOL, revision=REVISIONS[SMOL], dtype=DTYPE).to(DEVICE).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {n_params / 1e6:.0f} млн, весов в fp32: {n_params * 4 / 2**30:.2f} ГБ")
''')

replace(8, '''
def ask(images, question, longest_edge=1024, max_new_tokens=128, history=None,
        mdl=None, proc=None, show_prompt=False, vision_cache=None):
    mdl = model if mdl is None else mdl
    proc = processor if proc is None else proc
    synchronize(mdl)
    started = time.perf_counter()
    messages, all_images = [], []
    for msg, imgs in history or []:
        messages.append(msg)
        all_images += imgs
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question}]
    messages.append({"role": "user", "content": content})
    all_images += images
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if show_prompt:
        print(text[:400], "...\\n")
    kwargs = {"images": all_images, "size": {"longest_edge": longest_edge}} if all_images else {}
    inputs = proc(text=text, return_tensors="pt", **kwargs).to(mdl.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=DTYPE)
    synchronize(mdl)
    generation_started = time.perf_counter()
    vision_reused = False
    with torch.inference_mode():
        if vision_cache is not None and all_images:
            signature = (id(mdl), longest_edge, tuple(hashlib.sha256(image.tobytes()).hexdigest() for image in all_images))
            vision_reused = vision_cache.get("signature") == signature
            if not vision_reused:
                vision_cache.clear()
                vision_cache["signature"] = signature
                vision_cache["outputs"] = {"image": mdl.get_image_features(
                    pixel_values=inputs["pixel_values"],
                    pixel_attention_mask=inputs.get("pixel_attention_mask"), return_dict=True)}
            inputs.pop("pixel_values", None)
            inputs.pop("pixel_attention_mask", None)
            inputs["mm_encoder_outputs"] = vision_cache["outputs"]
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    synchronize(mdl)
    generation_seconds = time.perf_counter() - generation_started
    answer = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    image_token = proc.tokenizer.convert_tokens_to_ids("<image>")
    info = {"visual_tokens": int((inputs.input_ids == image_token).sum().item()),
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "seconds": time.perf_counter() - started, "generation_seconds": generation_seconds,
            "vision_reused": vision_reused}
    return answer, info
''')

replace(10, '''
from io import BytesIO

import requests
from PIL import Image

url = "https://images.cocodataset.org/val2017/000000039769.jpg"
image_path = Path("artifacts/cats_original.jpg")
if image_path.exists():
    cats = Image.open(image_path).convert("RGB")
else:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    cats = Image.open(BytesIO(response.content)).convert("RGB")
display(cats.resize((cats.width // 2, cats.height // 2)))
answer, info = ask([cats], "Describe the image in one sentence.", show_prompt=True)
print("Ответ:", answer)
print(info)
''')

source = "".join(nb["cells"][14]["source"])
source = source.replace("from PIL import ImageDraw, ImageFont", "from PIL import ImageDraw, ImageFont\nimport matplotlib")
source = source.replace('"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]',
                        'str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"),\n                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]')
replace(14, source)

source = "".join(nb["cells"][17]["source"])
start = source.index("def extract_by_fields")
end = source.index("def levenshtein")
source = source[:start] + '''def extract_by_fields(image, longest_edge=2048, mdl=None, proc=None, return_info=False, reuse_vision=True):
    pred, details = {}, {}
    vision_cache = {} if reuse_vision else None
    for field, question in FIELD_QUESTIONS.items():
        pred[field], details[field] = ask([image], question, longest_edge=longest_edge,
                                        max_new_tokens=24, mdl=mdl, proc=proc, vision_cache=vision_cache)
    return (pred, details) if return_info else pred


''' + source[end:]
source = source.replace('"anls": round(anls1(p, truth[k]), 2)', '"anls": anls1(p, truth[k])')
replace(17, source)

replace(22, '''
def run_resolution_experiment(name, mdl, proc, rows):
    for edge in [512, 1024, 2048]:
        pred, details = extract_by_fields(invoice, longest_edge=edge, mdl=mdl, proc=proc, return_info=True)
        res = check(pred, truth, post=True)
        rows.append({"model": name, "longest_edge": edge,
                     "visual_tokens": details["invoice_number"]["visual_tokens"],
                     "seconds (4 вопроса)": sum(item["seconds"] for item in details.values()),
                     "exact (из 4)": sum(r["exact"] for r in res),
                     "ANLS": sum(r["anls"] for r in res) / 4,
                     "invoice_number": res[0]["pred"], "seller": res[2]["pred"]})

rows = []
run_resolution_experiment("256M", model, processor, rows)
model.to("cpu")
if DEVICE == "cuda":
    torch.cuda.empty_cache()
proc_500 = AutoProcessor.from_pretrained(SMOL_500, revision=REVISIONS[SMOL_500])
model_500 = AutoModelForImageTextToText.from_pretrained(SMOL_500, revision=REVISIONS[SMOL_500], dtype=DTYPE).to(DEVICE).eval()
run_resolution_experiment("500M", model_500, proc_500, rows)
results = pd.DataFrame(rows)
display(results)
''')

replace(24, '''
del model_500, proc_500
activate_model(model)
print(f"Память GPU: {torch.cuda.memory_allocated() / 2**30:.2f} ГБ" if DEVICE == "cuda" else "CPU")
''')

source = "".join(nb["cells"][29]["source"])
source = source.replace('flor_proc = AutoProcessor.from_pretrained(FLOR)',
                        'model.to("cpu")\nif DEVICE == "cuda":\n    torch.cuda.empty_cache()\nflor_proc = AutoProcessor.from_pretrained(FLOR, revision=REVISIONS[FLOR])')
source = source.replace('from_pretrained(FLOR, dtype=DTYPE)', 'from_pretrained(FLOR, revision=REVISIONS[FLOR], dtype=DTYPE)')
source = source.replace('def florence(task, image, text_input=""):', 'def florence(task, image, text_input="", max_new_tokens=256):')
source = source.replace('.to(DEVICE)\n    with torch.inference_mode():', '.to(flor.device)\n    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=DTYPE)\n    with torch.inference_mode():')
source = source.replace('max_new_tokens=256, num_beams=3', 'max_new_tokens=max_new_tokens, num_beams=3')
replace(29, source)

replace(36, '''
results.to_csv(RESULTS_DIR / "demo_resolution_experiment.csv", index=False)
SAVE_TO_DRIVE = False
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    results.to_csv("/content/drive/MyDrive/vlm_resolution_experiment.csv", index=False)
print(RESULTS_DIR.resolve())
''')

nb["cells"][0]["source"] = ("".join(nb["cells"][0]["source"])
    .replace("а ответы — такими же (генерация детерминированная)",
             "возможны небольшие отличия ответов между версиями библиотек и устройствами даже при жадной генерации")
    + "\n\n**Расширение практикума:** решения всех пяти заданий расположены после исходного раздела 11. "
    "Сохранённые выше демонстрационные выводы относятся к исходному авторскому прогону, пока ячейки не выполнены повторно. "
    "Продолжительность полного расширенного эксперимента зависит от устройства и существенно превышает время исходных демонстраций.\n").splitlines(keepends=True)
nb["cells"][7]["source"].append("\n\nВремя теперь измеряется с синхронизацией CUDA; отдельно сохраняются полное время вызова и время генерации. `history` содержит пары (сообщение, список изображений), `images` — изображения текущего вопроса.\n")
nb["cells"][18]["source"].append("\n\nANLS рассчитывается исходной функцией `anls1`; округление удалено из `check`, чтобы оно не искажало средние метрики. Округлять можно только отображение таблиц.\n")
nb["cells"][35]["source"].append("\n\nВсе CSV автоматически сохраняются локально в `vlm_results`. Подключение Google Диска необязательно: установите `SAVE_TO_DRIVE = True` в следующей ячейке, если нужна интерактивная авторизация. Для копирования всех итоговых файлов после экспериментов предусмотрена последняя ячейка.\n")

add("markdown", '''
### 11.0. Общий протокол самостоятельных экспериментов

Используем настоящие вызовы моделей в `float32`, `do_sample=False`, `num_beams=1` для SmolVLM и `num_beams=3` для Florence. Счета создаются исходной `make_invoice` для **всех seed от 0 до 19**, с размером шрифта **14**, как в демонстрации раздела 6. Это одна семья синтетических шаблонов; результат нельзя переносить на произвольные реальные документы. Таблицы содержат исходный ответ, обработанное значение, эталон, Exact Match и ANLS; Exact Match использует исходные `check` и `postprocess`, а ANLS — исходную `anls1` без промежуточного округления.

Кэш хранит только реальные ответы после завершённого вызова модели. Ключ включает пиксели изображения, модель и её ревизию, вопрос, разрешение, параметры генерации, версии библиотек и устройство; эталон не передаётся модели. Повторное использование одного и того же предсказания явно отмечается в CSV. Сохранённое время означает время первоначального вычисления, а не чтения кэша; для нового замера задайте `USE_CACHE = False`.

Внутри `extract_by_fields` визуальный энкодер кодирует страницу один раз; исходные визуальные признаки через штатный `mm_encoder_outputs` передаются в четыре независимых вызова генерации. Кэш признаков живёт только в пределах одного документа и разрешения, а флаг `vision_reused` показывает повторное использование; для проверки эквивалентности доступен `reuse_vision=False`. Время четырёх вопросов включает одно реальное кодирование страницы и все четыре декодирования, поэтому его нельзя напрямую сопоставлять со старым авторским замером без этой оптимизации.

Каждое задание сохраняет отдельные CSV в `vlm_results`; ошибки загрузки или инференса прерывают выполнение и не заменяются предсказаниями. Исходные демонстрации и их последовательность сохранены. Проверенные API: [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), [Florence-2](https://huggingface.co/florence-community/Florence-2-base), [Transformers 5.22.1](https://pypi.org/project/transformers/5.22.1/).
''', "assignment-protocol")

add("code", '''
from datetime import date
import inspect
import matplotlib.pyplot as plt

USE_CACHE = True
CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
FIELDS = list(FIELD_QUESTIONS)
DOCUMENT_SEEDS = list(range(20))
FONT_SIZE = 14
DOCUMENTS = {seed: make_invoice(seed=seed, font_size=FONT_SIZE) for seed in DOCUMENT_SEEDS}
PROTOCOL_VERSION = "vlm-practicum-2-vision-reuse"

def save_table(frame, name):
    frame.to_csv(RESULTS_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.max_rows", 100, "display.max_columns", 40, "display.max_colwidth", 100):
        display(frame)

def image_digest(image):
    header = f"{image.mode}:{image.size}".encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()

def cache_request(image, mdl, proc, request):
    payload = {"protocol": PROTOCOL_VERSION, "image_sha256": image_digest(image),
               "model": mdl.config._name_or_path,
               "revision": getattr(mdl.config, "_commit_hash", None),
               "dtype": str(DTYPE), "device": DEVICE, "gpu": ENVIRONMENT["gpu"],
               "versions": VERSIONS, "processor": type(proc).__name__,
               "image_processor": proc.image_processor.to_dict(), "request": request}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json", payload

def read_cached(path, payload):
    if USE_CACHE and path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["request"] == json.loads(json.dumps(payload, default=str)):
            return record["result"]
    return None

def write_cached(path, payload, result):
    record = {"request": payload, "result": result}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)

def cached_question(image, question, edge, max_tokens=24, mdl=None, proc=None):
    mdl = model if mdl is None else mdl
    proc = processor if proc is None else proc
    path, payload = cache_request(image, mdl, proc,
        {"question": question, "edge": edge, "max_new_tokens": max_tokens,
         "do_sample": False, "num_beams": 1})
    record = read_cached(path, payload)
    reused = record is not None
    if record is None:
        answer, info = ask([image], question, longest_edge=edge,
                           max_new_tokens=max_tokens, mdl=mdl, proc=proc)
        record = {"answer": answer, "info": info}
        write_cached(path, payload, record)
    return record["answer"], record["info"], reused

def document_experiment(seed, edge):
    image, gold = DOCUMENTS[seed]
    path, payload = cache_request(image, model, processor,
        {"questions": FIELD_QUESTIONS, "edge": edge, "max_new_tokens": 24,
         "do_sample": False, "num_beams": 1})
    record = read_cached(path, payload)
    reused = record is not None
    if record is None:
        prediction, details = extract_by_fields(image, longest_edge=edge, return_info=True)
        record = {"prediction": prediction, "details": details}
        write_cached(path, payload, record)
    measured = check(record["prediction"], gold, post=True)
    output = []
    for row in measured:
        field = row["field"]
        output.append({"seed": seed, "longest_edge": edge, **row,
                       "raw_answer": record["prediction"][field],
                       **record["details"][field], "reused": reused})
    return output

def summarize_documents(fields):
    return fields.groupby(["longest_edge", "seed"], as_index=False).agg(
        correct_fields=("exact", "sum"), field_count=("exact", "size"),
        exact_accuracy=("exact", "mean"), anls=("anls", "mean"),
        visual_tokens_per_question=("visual_tokens", "first"),
        visual_tokens_all_questions=("visual_tokens", "sum"),
        seconds=("seconds", "sum"), generation_seconds=("generation_seconds", "sum"),
        reused=("reused", "all"))

def summarize_resolutions(fields):
    docs = summarize_documents(fields)
    summary = docs.groupby("longest_edge", as_index=False).agg(
        documents=("seed", "nunique"), fields=("field_count", "sum"),
        correct_fields=("correct_fields", "sum"), exact_accuracy=("exact_accuracy", "mean"),
        anls=("anls", "mean"), visual_tokens=("visual_tokens_per_question", "mean"),
        seconds_per_document=("seconds", "mean"), total_seconds=("seconds", "sum"))
    per_field = fields.pivot_table(index="longest_edge", columns="field", values="exact", aggfunc="mean")
    per_field = per_field.rename(columns=lambda name: f"exact_{name}").reset_index()
    return summary.merge(per_field, on="longest_edge", validate="one_to_one")

manifest = pd.DataFrame([{"seed": seed, "font_size": FONT_SIZE, "width": image.width,
                         "height": image.height, "image_sha256": image_digest(image), **gold}
                        for seed, (image, gold) in DOCUMENTS.items()])
save_table(manifest, "documents_manifest")
activate_model(model)
''', "assignment-common-code")

add("markdown", '''
### 11.1. Задание 1 — исследование галлюцинаций

Эталоны составлены после визуального просмотра исходного COCO-изображения `000000039769.jpg`, до получения ответов моделей. На розовом диване лежат две кошки и два пульта; отчётливо видны лапы и хвост. На изображении нет собаки, велосипеда, книги, чашки и ноутбука. Пять положительных вопросов включают части животного (`paw`, `tail`): это явно присутствующие объекты, но зависимые вопросы, а не пять независимых классов POPE.

Обе модели получают одинаковые вопросы, `longest_edge=1024` и `max_new_tokens=24`. Нормализатор принимает только однозначный ответ `yes`/`no` в начале, допускает короткое пояснение, но отклоняет ответы с обеими полярностями, словами неопределённости или противоречием. Любой `ambiguous` считается ошибкой accuracy; FPR = число ответов `yes` на пять отсутствующих объектов / 5. Доля неоднозначных ответов публикуется отдельно, поэтому отказ отвечать не выглядит высокой точностью.
''', "assignment-1")

add("code", r'''
HALLUCINATION_ITEMS = [
    ("cat", "yes", "Две кошки занимают левую и правую части кадра"),
    ("remote control", "yes", "Два пульта лежат между кошками и над левой кошкой"),
    ("couch", "yes", "Розовое сиденье и спинка дивана"),
    ("paw", "yes", "Лапы обеих кошек видны на сиденье"),
    ("tail", "yes", "Хвост левой кошки виден внизу слева"),
    ("dog", "no", "Собаки в кадре нет"),
    ("bicycle", "no", "Велосипеда в кадре нет"),
    ("book", "no", "Книги в кадре нет"),
    ("cup", "no", "Чашки в кадре нет"),
    ("laptop", "no", "Ноутбука в кадре нет"),
]

def normalize_yes_no(answer):
    text = str(answer).strip().lower().replace("’", "'")
    text = re.sub(r"^[\s*'\"]+", "", text)
    tokens = re.findall(r"\b(?:yes|no)\b", text)
    first = re.match(r"^(yes|no)\b", text)
    uncertain = re.search(r"\b(maybe|perhaps|possibly|probably|uncertain|unsure|might|cannot|can't|could|unclear)\b|not sure|unable to", text)
    if first is None or len(set(tokens)) != 1 or uncertain:
        return "ambiguous"
    result = first.group(1)
    if result == "yes" and re.search(r"\b(not|no|isn't|aren't|doesn't|without)\b", text):
        return "ambiguous"
    if result == "no" and re.search(r"\bthere (is|are)\b(?!\s+(no|not)\b)", text):
        return "ambiguous"
    return result

hallucination_gold = pd.DataFrame([{"object": obj, "question": f"Is there a {obj} in the image?",
                                   "gold": gold, "visual_evidence": evidence}
                                  for obj, gold, evidence in HALLUCINATION_ITEMS])
display(cats)
save_table(hallucination_gold, "task1_visual_ground_truth")
hallucination_rows = []

def evaluate_hallucinations(label, mdl, proc):
    activate_model(mdl)
    for row in hallucination_gold.to_dict("records"):
        answer, info, reused = cached_question(cats, row["question"], 1024, mdl=mdl, proc=proc)
        normalized = normalize_yes_no(answer)
        hallucination_rows.append({**row, "model": label, "raw_answer": answer,
                                    "normalized": normalized, "correct": normalized == row["gold"],
                                    **info, "reused": reused})

evaluate_hallucinations("256M", model, processor)
model.to("cpu")
if DEVICE == "cuda":
    torch.cuda.empty_cache()
proc_500 = AutoProcessor.from_pretrained(SMOL_500, revision=REVISIONS[SMOL_500])
model_500 = AutoModelForImageTextToText.from_pretrained(SMOL_500, revision=REVISIONS[SMOL_500], dtype=DTYPE).to(DEVICE).eval()
evaluate_hallucinations("500M", model_500, proc_500)
del model_500, proc_500
activate_model(model)

task1_answers = pd.DataFrame(hallucination_rows)
task1_comparison = hallucination_gold.copy()
for label in ["256M", "500M"]:
    subset = task1_answers.loc[task1_answers.model.eq(label), ["question", "raw_answer", "normalized", "correct"]]
    subset = subset.rename(columns={name: f"{name}_{label}" for name in ["raw_answer", "normalized", "correct"]})
    task1_comparison = task1_comparison.merge(subset, on="question", validate="one_to_one")
task1_metrics = []
for label, group in task1_answers.groupby("model"):
    absent = group.loc[group.gold.eq("no")]
    task1_metrics.append({"model": label, "questions": len(group), "accuracy": group.correct.mean(),
                          "absent_objects": len(absent), "false_yes": int(absent.normalized.eq("yes").sum()),
                          "false_positive_rate": absent.normalized.eq("yes").mean(),
                          "yes_rate": group.normalized.eq("yes").mean(),
                          "ambiguous": int(group.normalized.eq("ambiguous").sum())})
task1_metrics = pd.DataFrame(task1_metrics)
task1_answers.to_csv(RESULTS_DIR / "task1_answers_long.csv", index=False, encoding="utf-8-sig")
save_table(task1_comparison, "task1_comparison")
save_table(task1_metrics, "task1_metrics")
''', "assignment-1-code")

add("markdown", '''
#### Выводы к заданию 1

Новые измерения обеих моделей пока не выполнены; сохранённые демонстрации выше не являются результатами этих десяти вопросов. Accuracy, доля ложных `yes` и число неоднозначных ответов будут вычислены в таблице непосредственно перед этим разделом. Без этих измерений нельзя утверждать, что 500M галлюцинирует реже 256M. Даже после выполнения вывод относится к одному изображению и десяти зависимым вопросам.
''', "assignment-1-conclusion")

add("markdown", '''
### 11.2. Задание 2 — устойчивость извлечения данных

SmolVLM-256M извлекает четыре поля из каждого из 20 счетов при `longest_edge=1024` и `2048`: всего 40 конфигураций документов и 160 ответов. Сначала сохраняется необработанный ответ, затем применяются `postprocess`, `check` и `anls1`. Общая accuracy — число правильных полей / 80 для каждого разрешения; дополнительно приводятся accuracy и ANLS по каждому полю, а также результаты каждого документа. Частые ошибки определяются по фактическому числу несовпадений, без догадок об их причинах.
''', "assignment-2")

add("code", '''
activate_model(model)
task2_records = []
for edge in [1024, 2048]:
    for seed in DOCUMENT_SEEDS:
        task2_records.extend(document_experiment(seed, edge))
        print(f"Задание 2: {edge}px, документ {seed + 1}/20", flush=True)
task2_fields = pd.DataFrame(task2_records)
assert len(task2_fields) == 160
assert not task2_fields.duplicated(["seed", "longest_edge", "field"]).any()
task2_documents = summarize_documents(task2_fields)
task2_summary = summarize_resolutions(task2_fields)
task2_field_metrics = task2_fields.groupby(["longest_edge", "field"], as_index=False).agg(
    count=("exact", "size"), correct=("exact", "sum"), exact_accuracy=("exact", "mean"), anls=("anls", "mean"))
task2_field_metrics["errors"] = task2_field_metrics["count"] - task2_field_metrics["correct"]
task2_field_metrics = task2_field_metrics.sort_values(["longest_edge", "errors", "field"], ascending=[True, False, True])
save_table(task2_fields, "task2_fields")
save_table(task2_documents, "task2_documents")
save_table(task2_summary, "task2_summary")
save_table(task2_field_metrics, "task2_field_metrics")
''', "assignment-2-code")

add("markdown", '''
#### Выводы к заданию 2

Извлечение 160 полей на полной выборке пока не выполнено, поэтому численных выводов о точности нет. После запуска сравнение разрешений и список наиболее ошибочных полей определяются таблицами `task2_summary` и `task2_field_metrics`. Эталоны создаются исходным генератором и никогда не передаются модели в подсказке. Даже измеренное преимущество одного разрешения будет характеризовать только этот шаблон счёта и выбранный шрифт.
''', "assignment-2-conclusion")

add("markdown", '''
### 11.3. Задание 3 — кривая разрешения

Расширяем эксперимент раздела 7 до `512, 768, 1024, 1536, 2048` на **тех же 20 документах**, используя SmolVLM-256M. Результаты 1024 и 2048 берутся из задания 2; остальные разрешения вычисляются заново. Число визуальных токенов подсчитывается непосредственно по `<image>` в фактическом входе каждого вопроса; на графике указаны токены одного вопроса, а в CSV также сохранена сумма по четырём вопросам. Время включает подготовку входа, передачу на устройство, генерацию и декодирование; загрузка весов не включена.

«Колено» ищется только как описательная эвристика: кривая должна быть неубывающей, иметь прирост не менее 0,10, внутреннюю точку с отклонением от нормированной диагонали не менее 0,10 и снижением среднего наклона после неё минимум вдвое; остаточный прирост после неё — не более 0,05. Одинаковые количества токенов объединяются средним значением accuracy только для этой проверки. Если условия не выполнены, выводится «не обнаружено»; критерий не доказывает статистическую значимость. Выборка из 20 синтетических документов одного шаблона ограничивает репрезентативность.
''', "assignment-3")

add("code", '''
task3_records = task2_fields.to_dict("records")
for row in task3_records:
    row["reused"] = True
for edge in [512, 768, 1536]:
    for seed in DOCUMENT_SEEDS:
        task3_records.extend(document_experiment(seed, edge))
        print(f"Задание 3: {edge}px, документ {seed + 1}/20", flush=True)
task3_fields = pd.DataFrame(task3_records).sort_values(["longest_edge", "seed", "field"])
assert len(task3_fields) == 400
assert not task3_fields.duplicated(["seed", "longest_edge", "field"]).any()
task3_documents = summarize_documents(task3_fields)
task3_summary = summarize_resolutions(task3_fields)

def find_knee(summary):
    points = summary.groupby("visual_tokens", as_index=False).exact_accuracy.mean().sort_values("visual_tokens")
    x = points.visual_tokens.to_numpy(dtype=float)
    y = points.exact_accuracy.to_numpy(dtype=float)
    if len(x) < 3 or np.any(np.diff(y) < -1e-12) or y[-1] - y[0] < 0.10:
        return {"detected": False, "visual_tokens": None, "resolution": None, "reason": "Нет достаточного монотонного прироста"}
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y[0]) / (y[-1] - y[0])
    index = int(np.argmax((yn - xn)[1:-1])) + 1
    before = (y[index] - y[0]) / (x[index] - x[0])
    after = (y[-1] - y[index]) / (x[-1] - x[index])
    detected = bool(yn[index] - xn[index] >= 0.10 and before > 0 and after <= before / 2
                    and y[-1] - y[index] <= 0.05 + 1e-12)
    edges = summary.loc[summary.visual_tokens.eq(x[index]), "longest_edge"].astype(str).tolist()
    return {"detected": detected, "visual_tokens": int(x[index]) if detected else None,
            "resolution": ", ".join(edges) if detected else None,
            "reason": "Эвристическое колено" if detected else "Нет выраженного насыщения по заданному критерию"}

task3_knee = find_knee(task3_summary)
save_table(task3_documents, "task3_documents")
save_table(task3_summary, "task3_summary")
save_table(pd.DataFrame([task3_knee]), "task3_knee")
task3_fields.to_csv(RESULTS_DIR / "task3_fields.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(task3_summary.visual_tokens, task3_summary.exact_accuracy, marker="o", label="Exact Match")
for index, row in task3_summary.iterrows():
    axes[0].annotate(f"{int(row.longest_edge)} px", (row.visual_tokens, row.exact_accuracy),
                     xytext=(5, 8 + 12 * (index % 2)), textcoords="offset points", fontsize=9)
if task3_knee["detected"]:
    axes[0].axvline(task3_knee["visual_tokens"], linestyle="--", color="tab:red", label="Knee heuristic")
axes[0].set(xlabel="Visual tokens per question", ylabel="Correct fields / all fields", ylim=(-0.03, 1.15))
axes[0].legend()
axes[1].plot(task3_summary.visual_tokens, task3_summary.seconds_per_document, marker="o")
axes[1].set(xlabel="Visual tokens per question", ylabel="Seconds per document (4 questions)")
for axis in axes:
    axis.grid(alpha=0.3)
fig.suptitle("SmolVLM-256M: 20 synthetic invoices, float32")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "task3_resolution_curve.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)
''', "assignment-3-code")

add("markdown", '''
#### Выводы к заданию 3

Новая кривая по пяти разрешениям пока не измерена; график и положение возможного «колена» не подставлены из демонстрационного прогона. После выполнения таблица покажет фактический компромисс между accuracy, ANLS, числом токенов и временем. Эвристика имеет право вернуть отсутствие выраженного «колена», в том числе при немонотонной точности. Все выводы будут ограничены двадцатью синтетическими счетами одного шаблона.
''', "assignment-3-conclusion")

add("markdown", '''
### 11.4. Задание 4 — программная проверка без эталона

Единица основного анализа — **одно поле** из 160 ответов задания 2. Исходная `validate` вызывается для каждого предсказанного документа; её сообщения сопоставляются с конкретными полями. Новая `validate_extended` сохраняет исходные проверки и добавляет календарную корректность даты через `date.fromisoformat`, а также проверку непустого продавца; эталон в функции валидации не передаётся.

Только затем флаг отклонения сравнивается с `exact` из задания 2: TP — ошибочное и отклонённое поле, FN — ошибочное, но принятое, FP — правильное, но отклонённое, TN — правильное и принятое. Доля обнаруженных ошибок = TP / (TP + FN); при отсутствии ошибок она не определена (`NaN`), а не равна 100%. Дополнительно приводится анализ на уровне документов: документ ошибочен, если неверно хотя бы одно поле. Проверка формата не способна распознать правдоподобную, но неверную дату или корректный по форме выдуманный номер.
''', "assignment-4")

add("code", r'''
ISSUE_FIELDS = {
    "номер не в формате INV-NNNNN": "invoice_number",
    "дата не в формате YYYY-MM-DD": "date",
    "итог — не целое число": "total",
    "несуществующая календарная дата": "date",
    "продавец не указан": "seller",
}

def validate_extended(pred):
    problems = [problem for problem in validate(pred) if problem != "ок"]
    cleaned = {field: postprocess(field, pred.get(field, "")) for field in FIELDS}
    value = cleaned["date"]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            date.fromisoformat(value)
        except ValueError:
            problems.append("несуществующая календарная дата")
    if not cleaned["seller"].strip():
        problems.append("продавец не указан")
    return problems or ["ок"]

task4_records = []
for (edge, seed), group in task2_fields.groupby(["longest_edge", "seed"]):
    prediction = dict(zip(group.field, group.raw_answer))
    for variant, validator in [("original", validate), ("extended", validate_extended)]:
        issues = [issue for issue in validator(prediction) if issue != "ок"]
        unknown = set(issues) - set(ISSUE_FIELDS)
        if unknown:
            raise ValueError(f"Неизвестные сообщения валидатора: {unknown}")
        flagged_fields = {ISSUE_FIELDS[issue] for issue in issues}
        for row in group.to_dict("records"):
            error = not bool(row["exact"])
            flagged = row["field"] in flagged_fields
            category = "TP" if error and flagged else "FN" if error else "FP" if flagged else "TN"
            task4_records.append({"variant": variant, "longest_edge": edge, "seed": seed,
                                  "field": row["field"], "raw_answer": row["raw_answer"],
                                  "pred": row["pred"], "gold": row["gold"],
                                  "actual_error": error, "flagged": flagged, "category": category,
                                  "issues": "; ".join(issue for issue in issues if ISSUE_FIELDS[issue] == row["field"])})
task4_fields = pd.DataFrame(task4_records)

def validation_metrics(group):
    error, flagged = group.actual_error, group.flagged
    tp = int((error & flagged).sum())
    fn = int((error & ~flagged).sum())
    fp = int((~error & flagged).sum())
    tn = int((~error & ~flagged).sum())
    return {"predictions": len(group), "actual_errors": tp + fn, "flagged": tp + fp,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "error_detection_rate": tp / (tp + fn) if tp + fn else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan")}

def validation_summary(frame, unit):
    output = []
    for variant, group in frame.groupby("variant"):
        output.append({"variant": variant, "longest_edge": "all", "unit": unit, **validation_metrics(group)})
        for edge, subset in group.groupby("longest_edge"):
            output.append({"variant": variant, "longest_edge": str(edge), "unit": unit, **validation_metrics(subset)})
    return pd.DataFrame(output)

task4_documents = task4_fields.groupby(["variant", "longest_edge", "seed"], as_index=False).agg(
    actual_error=("actual_error", "any"), flagged=("flagged", "any"))
task4_summary = pd.concat([validation_summary(task4_fields, "field"),
                           validation_summary(task4_documents, "document")], ignore_index=True)
assert (task4_summary.TP + task4_summary.FN == task4_summary.actual_errors).all()
assert (task4_summary.TP + task4_summary.FN + task4_summary.FP + task4_summary.TN == task4_summary.predictions).all()
save_table(task4_fields, "task4_fields")
save_table(task4_documents, "task4_documents")
save_table(task4_summary, "task4_summary")
''', "assignment-4-code")

add("markdown", '''
#### Выводы к заданию 4

Без завершённого задания 2 нельзя измерить количество реальных ошибок или долю, обнаруженную валидатором. Новая календарная проверка обнаруживает невозможные даты, но не гарантирует улучшения на конкретных предсказаниях модели. В итоговой таблице отклонённые правильные значения учитываются отдельно как ложные срабатывания. Форматно корректные несовпадения с эталоном остаются пропусками проверки и не считаются обнаруженными ошибками.
''', "assignment-4-conclusion")

add("markdown", '''
### 11.5. Задание 5 — гибрид OCR + VLM

Сравниваем номер счёта на **всех тех же 20 документах**. Обоим подходам передаётся одна и та же копия страницы, заранее уменьшенная до длинной стороны **512 пикселей** методом LANCZOS; SmolVLM использует `longest_edge=512`. Florence получает эту же уменьшенную копию, а не исходный документ: её стандартный процессор может масштабировать вход до собственного размера, но детали исходного высокого разрешения ему недоступны. Поэтому сравнение контролирует доступную визуальную информацию; внутренние размеры тензоров разных архитектур не обязаны совпадать.

Baseline получает только изображение и исходный вопрос о номере счёта. Hybrid получает то же изображение, тот же вопрос и текст Florence `<OCR>` в формате `Text on the page: ... What is the invoice number?`; эталон не добавляется. OCR сохраняется целиком; `max_new_tokens=512` для OCR и 24 для ответа SmolVLM, без стохастической генерации. Сначала выполняется OCR для всех страниц, затем Florence освобождает GPU и обе серии SmolVLM выполняются по документам. Отчёт показывает парные переходы «исправлен» / «испорчен», разницу Exact Match в процентных пунктах и разницу ANLS; улучшение не предполагается заранее.
''', "assignment-5")

add("code", '''
LOW_EDGE = 512
LOW_DOCUMENTS = {}
for seed, (image, gold) in DOCUMENTS.items():
    scaled = image.copy()
    scaled.thumbnail((LOW_EDGE, LOW_EDGE), Image.Resampling.LANCZOS)
    LOW_DOCUMENTS[seed] = scaled

def cached_ocr(image):
    path, payload = cache_request(image, flor, flor_proc,
        {"task": "<OCR>", "max_new_tokens": 512, "do_sample": False, "num_beams": 3})
    record = read_cached(path, payload)
    reused = record is not None
    if record is None:
        synchronize(flor)
        started = time.perf_counter()
        text = florence("<OCR>", image, max_new_tokens=512)
        synchronize(flor)
        if not isinstance(text, str):
            raise TypeError(f"OCR должен возвращать строку, получен {type(text)}")
        record = {"text": text, "seconds": time.perf_counter() - started}
        write_cached(path, payload, record)
    return record, reused

activate_model(flor)
ocr_records = []
for seed in DOCUMENT_SEEDS:
    record, reused = cached_ocr(LOW_DOCUMENTS[seed])
    ocr_records.append({"seed": seed, "ocr_text": record["text"], "ocr_seconds": record["seconds"],
                        "reused": reused, "width": LOW_DOCUMENTS[seed].width,
                        "height": LOW_DOCUMENTS[seed].height})
    print(f"Задание 5, OCR: документ {seed + 1}/20", flush=True)
task5_ocr = pd.DataFrame(ocr_records)
save_table(task5_ocr, "task5_ocr")
activate_model(model)
task5_records = []
for row in task5_ocr.to_dict("records"):
    seed = row["seed"]
    image = LOW_DOCUMENTS[seed]
    gold = DOCUMENTS[seed][1]["invoice_number"]
    question = FIELD_QUESTIONS["invoice_number"]
    prompts = {"Baseline": question, "Hybrid": f"Text on the page: {row['ocr_text']}\\n\\n{question}"}
    for approach, prompt in prompts.items():
        raw, info, reused = cached_question(image, prompt, LOW_EDGE)
        prediction = postprocess("invoice_number", raw)
        task5_records.append({"seed": seed, "approach": approach, "longest_edge": LOW_EDGE,
                              "raw_answer": raw, "pred": prediction, "gold": gold,
                              "exact": prediction == str(gold), "anls": anls1(prediction, gold),
                              **info, "reused": reused,
                              "ocr_seconds": row["ocr_seconds"] if approach == "Hybrid" else 0.0})
    print(f"Задание 5, VLM: документ {seed + 1}/20", flush=True)
task5_fields = pd.DataFrame(task5_records)
assert len(task5_fields) == 40
task5_fields["total_seconds"] = task5_fields.seconds + task5_fields.ocr_seconds
task5_summary = task5_fields.groupby("approach", as_index=False).agg(
    documents=("seed", "nunique"), correct=("exact", "sum"), exact_accuracy=("exact", "mean"),
    anls=("anls", "mean"), seconds_per_document=("total_seconds", "mean"),
    visual_tokens=("visual_tokens", "mean"))
task5_comparison = task5_fields.loc[task5_fields.approach.eq("Baseline")].merge(
    task5_fields.loc[task5_fields.approach.eq("Hybrid")], on=["seed", "gold", "longest_edge"],
    suffixes=("_baseline", "_hybrid"), validate="one_to_one")
task5_comparison = task5_comparison[["seed", "gold", "raw_answer_baseline", "pred_baseline", "exact_baseline",
                                     "anls_baseline", "raw_answer_hybrid", "pred_hybrid", "exact_hybrid", "anls_hybrid"]]
indexed = task5_summary.set_index("approach")
task5_delta = pd.DataFrame([{
    "exact_gain_pp": 100 * (indexed.loc["Hybrid", "exact_accuracy"] - indexed.loc["Baseline", "exact_accuracy"]),
    "anls_gain": indexed.loc["Hybrid", "anls"] - indexed.loc["Baseline", "anls"],
    "fixed": int((~task5_comparison.exact_baseline & task5_comparison.exact_hybrid).sum()),
    "regressed": int((task5_comparison.exact_baseline & ~task5_comparison.exact_hybrid).sum()),
    "both_correct": int((task5_comparison.exact_baseline & task5_comparison.exact_hybrid).sum()),
    "both_wrong": int((~task5_comparison.exact_baseline & ~task5_comparison.exact_hybrid).sum()),
}])
task5_fields.to_csv(RESULTS_DIR / "task5_fields.csv", index=False, encoding="utf-8-sig")
save_table(task5_comparison, "task5_comparison")
save_table(task5_summary, "task5_summary")
save_table(task5_delta, "task5_delta")
''', "assignment-5-code")

add("markdown", '''
#### Выводы к заданию 5

Парное сравнение Baseline и Hybrid пока не выполнено, поэтому улучшение от OCR не заявляется. Разница accuracy будет вычислена в процентных пунктах, а ANLS — в исходной шкале от 0 до 1. Таблица парных переходов позволит увидеть, сколько номеров OCR-контекст исправил и сколько испортил. Даже при положительной разнице результат ограничен двадцатью документами одного шаблона при одинаковом низком разрешении исходной информации.
''', "assignment-5-conclusion")

add("markdown", '''
### 11.6. Сохранение всех результатов

Все таблицы, график и сведения об окружении уже записаны в `vlm_results`. Архив ниже включает CSV, график и кэш настоящих предсказаний, позволяющий продолжить прерванную работу с теми же настройками. Если подключён Google Диск, архив также копируется в `MyDrive`; иначе его можно скачать из панели файлов Colab. Статус локальной проверки и фактического выполнения записывается отдельно в заключительном разделе.
''', "assignment-export")

add("code", '''
import shutil

archive = shutil.make_archive("vlm_results", "zip", root_dir=RESULTS_DIR)
if SAVE_TO_DRIVE:
    shutil.copy2(archive, "/content/drive/MyDrive/vlm_results.zip")
print(archive)
''', "assignment-export-code")

add("markdown", '''
### Проверка реализации и фактическое выполнение

Статус будет уточнён после проверки в доступной локальной среде. Пустые выходы новых ячеек означают отсутствие выполненного эксперимента, а не нулевую точность. Численные результаты и выводы не заполняются предполагаемыми значениями.
''', "verification-status")

revisions_path = ROOT / "artifacts" / "model_revisions.json"
if revisions_path.exists():
    revisions = json.loads(revisions_path.read_text(encoding="utf-8"))
    source = "".join(nb["cells"][4]["source"])
    for name, revision in revisions.items():
        source = source.replace(f'"{name}": "main"', f'"{name}": "{revision}"')
    replace(4, source)

for index, cell in enumerate(nb["cells"]):
    cell.setdefault("id", f"original-{index:02d}")
    if cell["cell_type"] == "code":
        cell["source"] = clean("".join(cell["source"])).splitlines(keepends=True)
nb["nbformat_minor"] = 5
PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Updated {PATH.name}: {len(nb['cells'])} cells")

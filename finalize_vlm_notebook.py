import json
from pathlib import Path
import shutil
import sys
import time

import nbformat

from update_vlm_conclusions import update_conclusions


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "VLM_практикум_Colab.ipynb"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "vlm_results"
sys.stdout.reconfigure(encoding="utf-8")
status = json.loads((ARTIFACTS / "execution_status.json").read_text(encoding="utf-8"))
verification = json.loads((ARTIFACTS / "results_verification.json").read_text(encoding="utf-8"))
environment = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
assert status["full_execution"] and status["failure"] is None
assert verification["full_notebook_execution"]
nb = nbformat.read(PATH, as_version=4)
update_conclusions(nb, status["completed_cell_ids"])

intro = nb.cells[0].source
start = intro.index("> ℹ️")
nb.cells[0].source = intro[:start] + (
    f"> ℹ️ Текущие сохранённые выходы всех исполняемых ячеек получены при полном локальном запуске "
    f"{status['finished_utc'][:10]} на {environment['gpu'] or environment['device'].upper()}, "
    f"в точности {environment['dtype']}. Реальный запуск на T4 в этой среде не выполнялся; "
    "время и отдельные ответы могут отличаться между устройствами даже при жадной генерации.\n\n"
    "**Расширение практикума:** решения всех пяти заданий расположены после исходного раздела 11. "
    "Исходные теоретические материалы и демонстрации сохранены; авторские обсуждения прежнего прогона "
    "отмечены отдельно от новых результатов. Markdown-выводы к заданиям описывают сохранённый локальный "
    "прогон; при повторном запуске ориентируйтесь на заново вычисленные таблицы. Продолжительность "
    "полного расширенного эксперимента существенно превышает время исходных демонстраций.\n"
)
for index in [13, 23, 27, 33]:
    note = "> Историческое обсуждение из исходного практикума; конкретные ответы относятся к авторскому прогону. Ниже и выше сохранены новые реальные выходы локального запуска, которые могут отличаться.\n\n"
    if not nb.cells[index].source.startswith(note):
        nb.cells[index].source = note + nb.cells[index].source

nb.cells[39].source = nb.cells[39].source.replace("import inspect\n", "")
report = next(cell for cell in nb.cells if cell.id == "verification-status")
inference_path = ARTIFACTS / "inference_execution_status.json"
inference_note = ""
if inference_path.exists():
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    if inference["full_execution"]:
        inference_note = (
            f"Первый полный прогон с реальными вызовами нейросетей завершён {inference['finished_utc']}; "
            "при окончательном контрольном запуске задания повторно использовали проверенный кэш этих ответов. "
            "Времена в таблицах относятся к первоначальным вычислениям; журнал этого прогона сохранён "
            "в `artifacts/inference_execution_status.json`.\n\n"
        )
report.source = (
    "### Проверка реализации и фактическое выполнение\n\n"
    f"Полный последовательный запуск завершён **{status['finished_utc']}**: "
    f"выполнены все **{verification['executed_code_cells']} исполняемых ячеек**, включая исходные демонстрации "
    "и пять самостоятельных заданий, без ошибок исполнения. Использованы реальные веса SmolVLM-256M, "
    "SmolVLM-500M и Florence-2-base, фиксированные ревизии, `torch.float32` и `do_sample=False`. "
    f"Окружение: Python {environment['python']}, PyTorch {environment['versions']['torch']}, "
    f"Transformers {environment['versions']['transformers']}, устройство "
    f"**{environment['gpu'] or environment['device'].upper()}**; версии остальных библиотек сохранены "
    "в `vlm_results/environment.json`.\n\n"
    f"{inference_note}"
    "| Задание | Фактически проверенный объём |\n"
    "|---|---|\n"
    "| 1 | 10 вопросов × 2 модели = 20 настоящих ответов |\n"
    "| 2 | 20 счетов × 2 разрешения × 4 поля = 160 предсказаний |\n"
    "| 3 | 20 счетов × 5 разрешений × 4 поля = 400 результатов; 160 повторно использованы из задания 2 |\n"
    "| 4 | 160 полей и 40 конфигураций документов, исходная и расширенная валидация |\n"
    "| 5 | 20 результатов OCR и 40 ответов VLM: Baseline/Hybrid на тех же 20 счетах |\n\n"
    "Структура проверена через `nbformat.validate`, Python-синтаксис — через AST и `compile`. "
    "Во всех исполняемых ячейках отсутствуют комментарии, docstrings и отдельные поясняющие строки. "
    "Проверены воспроизводимость генератора, нормализация yes/no, ANLS, постобработка, календарная "
    "проверка, расчёт TP/FN/FP/TN и сохранение порядка исходных разделов. Метрики повторно рассчитаны "
    "по сохранённым предсказаниям, проверена их связь с кэшем вызовов; таблицы и PNG-график встроены "
    "в ноутбук, файлы CSV и график сохранены в `vlm_results`, архив — `vlm_results.zip`. "
    "Локальные проверки доступны в `verify_vlm_notebook.py` и `verify_vlm_results.py`, "
    "машинный отчёт — `artifacts/results_verification.json`.\n\n"
    "**Ограничения:** GPU T4 локально недоступна и не проверена; сохранённые времена относятся к CPU. "
    "Установка зависимостей и скачивание моделей в новом Colab-сеансе требуют интернета. "
    "В документных экспериментах используется двадцать счетов одного синтетического шаблона "
    "с мелким шрифтом, в проверке галлюцинаций — одна фотография с зависимыми вопросами; "
    "результаты не являются оценкой качества на произвольных реальных данных. "
    "Кэш содержит только завершённые реальные вызовы; для новых замеров времени отключите `USE_CACHE`.\n"
)
nbformat.validate(nb)
temporary = PATH.with_suffix(".tmp.ipynb")
nbformat.write(nb, temporary)
for attempt in range(20):
    try:
        temporary.replace(PATH)
        break
    except PermissionError:
        if attempt == 19:
            raise
        time.sleep(0.25)
shutil.make_archive(str(ROOT / "vlm_results"), "zip", root_dir=RESULTS)
print(f"Обновлён ноутбук: {PATH.name}; Markdown-выводы и отчёт основаны на завершённом прогоне.")

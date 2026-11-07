import json
from pathlib import Path
import re

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "VLM_практикум_Colab.ipynb"
RESULTS = ROOT / "vlm_results"


def update_conclusions(nb, completed=None):
    def put(number, text):
        target = f"assignment-{number}-conclusion"
        for cell in nb.cells:
            if cell.id == target:
                cell.source = f"#### Выводы к заданию {number}\n\n{text}"

    def allowed(number, name):
        return (completed is None or f"assignment-{number}-code" in completed) and (RESULTS / name).exists()

    if allowed(1, "task1_metrics.csv"):
        metrics = pd.read_csv(RESULTS / "task1_metrics.csv").set_index("model")
        small, large = metrics.loc["256M"], metrics.loc["500M"]
        direction = "выше" if large.accuracy > small.accuracy else "ниже" if large.accuracy < small.accuracy else "такая же"
        comparison = "как" if large.accuracy == small.accuracy else "чем"
        answers = pd.read_csv(RESULTS / "task1_answers_long.csv", keep_default_na=False)
        bed_error = answers.model.eq("500M") & answers.object.eq("cat") & answers.raw_answer.str.contains("on the bed", case=False)
        detail = ("В пояснении 500M к вопросу о кошке встречается «on the bed», хотя визуально это диван: "
                  "бинарная метрика оценивает только присутствие запрошенного объекта и не учитывает такую ошибку описания. "
                  if bed_error.any() else "")
        put(1, f"В фактическом локальном прогоне SmolVLM-256M получила accuracy {small.accuracy:.1%} "
            f"и долю ложных yes {small.false_positive_rate:.1%} на отсутствующих объектах. "
            f"У SmolVLM-500M эти показатели составили {large.accuracy:.1%} и {large.false_positive_rate:.1%}; "
            f"число неоднозначных ответов у 256M/500M — {int(small.ambiguous)}/{int(large.ambiguous)}. "
            f"Точность 500M на этом наборе {direction}, {comparison} у 256M, при абсолютной разнице {abs(large.accuracy-small.accuracy)*100:.1f} п.п. "
            f"{detail}"
            "Вывод ограничен одним изображением и десятью зависимыми вопросами, включая части животного.")

    if allowed(2, "task2_summary.csv"):
        metrics = pd.read_csv(RESULTS / "task2_summary.csv").set_index("longest_edge")
        low, high = metrics.loc[1024], metrics.loc[2048]
        fields = pd.read_csv(RESULTS / "task2_field_metrics.csv")
        worst = []
        for edge, group in fields.groupby("longest_edge"):
            errors = int(group.errors.max())
            names = ", ".join(group.loc[group.errors.eq(errors), "field"])
            worst.append(f"{edge}px — {names} ({errors}/20)" if errors else f"{edge}px — ошибок нет")
        predictions = pd.read_csv(RESULTS / "task2_fields.csv", keep_default_na=False)
        punctuation_errors = 0
        case_errors = int((~predictions.exact & predictions.pred.str.lower().eq(predictions.gold.str.lower())).sum())
        for row in predictions.loc[predictions.field.eq("invoice_number") & ~predictions.exact].itertuples():
            match = re.search(r"INV[./]\s*(\d+)", row.raw_answer, flags=re.I)
            if match and f"INV-{match.group(1)}" == row.gold:
                punctuation_errors += 1
        case_detail = (f"; число несовпадений только из-за регистра — {case_errors}, а исходная ANLS регистр игнорирует"
                       if case_errors else "")
        formatting = (f"В {punctuation_errors} ошибочных номерах цифры совпали с эталоном, но модель "
                      f"заменила дефис точкой или косой чертой{case_detail}. "
                      if punctuation_errors else "")
        put(2, f"На 20 счетах при 1024px верно распознано {int(low.correct_fields)}/80 полей "
            f"(Exact Match {low.exact_accuracy:.2%}, ANLS {low.anls:.4f}). "
            f"При 2048px результат составил {int(high.correct_fields)}/80 "
            f"(Exact Match {high.exact_accuracy:.2%}, ANLS {high.anls:.4f}), изменение точности — {(high.exact_accuracy-low.exact_accuracy)*100:+.1f} п.п. "
            f"Максимальное число ошибок по полям: {'; '.join(worst)}. "
            f"{formatting}"
            "Эти показатели характеризуют один синтетический шаблон с мелким шрифтом; причины ошибок требуют просмотра исходных ответов и не устанавливаются одной метрикой.")

    if allowed(3, "task3_summary.csv"):
        metrics = pd.read_csv(RESULTS / "task3_summary.csv").sort_values("longest_edge")
        environment = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
        device = environment["gpu"] or environment["device"].upper()
        first, last = metrics.iloc[0], metrics.iloc[-1]
        knee = pd.read_csv(RESULTS / "task3_knee.csv").iloc[0]
        knee_text = (f"Описательная эвристика обнаружила примерное колено при {int(knee.visual_tokens)} визуальных токенах "
                     f"(разрешение {knee.resolution}px)." if bool(knee.detected) else
                     "Выраженное колено по заранее указанному критерию не обнаружено.")
        put(3, f"На пяти разрешениях Exact Match находится в диапазоне {metrics.exact_accuracy.min():.2%}–{metrics.exact_accuracy.max():.2%}, "
            f"ANLS — {metrics.anls.min():.4f}–{metrics.anls.max():.4f}. "
            f"Между 512px и 2048px число визуальных токенов одного вопроса меняется с {int(first.visual_tokens)} до {int(last.visual_tokens)}, "
            f"а среднее время четырёх вопросов на {device} — с {first.seconds_per_document:.2f} до {last.seconds_per_document:.2f} с. "
            f"{knee_text} "
            "Результаты ограничены двадцатью документами одного шаблона; сохранённые времена относятся к первоначальным вычислениям, а не повторному чтению кэша.")

    if allowed(4, "task4_summary.csv"):
        metrics = pd.read_csv(RESULTS / "task4_summary.csv", dtype={"longest_edge": str})
        metrics = metrics.loc[metrics.unit.eq("field") & metrics.longest_edge.eq("all")].set_index("variant")
        base, extended = metrics.loc["original"], metrics.loc["extended"]
        rate = lambda value: f"{value:.1%}" if pd.notna(value) else "не определена из-за отсутствия ошибок"
        put(4, f"Среди 160 предсказанных полей найдено {int(base.actual_errors)} реальных ошибок; "
            f"исходная validate обнаружила {int(base.TP)} и пропустила {int(base.FN)}, ложных срабатываний — {int(base.FP)}. "
            f"После добавления календарной корректности и непустого продавца обнаружено {int(extended.TP)}, "
            f"пропущено {int(extended.FN)}, ложных срабатываний — {int(extended.FP)}. "
            f"Доля обнаруженных ошибок до расширения — {rate(base.error_detection_rate)}, после — {rate(extended.error_detection_rate)}; "
            f"дополнительно найдено {int(extended.TP-base.TP)} ошибок. "
            "Валидация не использовала эталоны: они применены только для измерения TP, FN, FP и TN, поэтому корректный формат не считается доказательством правильности значения.")

    if allowed(5, "task5_summary.csv"):
        metrics = pd.read_csv(RESULTS / "task5_summary.csv").set_index("approach")
        delta = pd.read_csv(RESULTS / "task5_delta.csv").iloc[0]
        base, hybrid = metrics.loc["Baseline"], metrics.loc["Hybrid"]
        change = "улучшилась" if delta.exact_gain_pp > 0 else "ухудшилась" if delta.exact_gain_pp < 0 else "не изменилась"
        put(5, f"При одинаковой исходной длинной стороне 512px Baseline верно извлёк {int(base.correct)}/20 номеров "
            f"(Exact Match {base.exact_accuracy:.1%}, ANLS {base.anls:.4f}). "
            f"Hybrid верно извлёк {int(hybrid.correct)}/20 номеров (Exact Match {hybrid.exact_accuracy:.1%}, ANLS {hybrid.anls:.4f}). "
            f"Точность {change}: изменение Exact Match {delta.exact_gain_pp:+.1f} п.п., ANLS {delta.anls_gain:+.4f}; "
            f"OCR-контекст исправил {int(delta.fixed)} ответов и испортил {int(delta.regressed)}. "
            "Это парное сравнение на двадцати документах одного шаблона, и дополнительный OCR также требует отдельного времени вычисления.")


if __name__ == "__main__":
    notebook = nbformat.read(PATH, as_version=4)
    status_path = ROOT / "artifacts" / "execution_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
    update_conclusions(notebook, status["completed_cell_ids"] if status else None)
    nbformat.write(notebook, PATH)

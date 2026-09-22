# Classical ML, pandas and NumPy

Все операции используют support-ticket dataset, который создаётся локально с `np.random.default_rng(42)`. По 200 записей на каждый из семи классов при default 1400 rows. Variation продукта, страны, тарифа и срочности даёт связанные поля priority/resolution_hours. Category определяется словарными формулировками: это намеренно лёгкий classification signal.

## Карта pandas

Основной файл — `src/ai_support_lab/classical_ml/dataset.py`.

| Операция | Функция / назначение |
|---|---|
| DataFrame | generate_dataset: сбор column arrays |
| read_csv | load_dataset: вход training data |
| dropna | удалить строки без title/category |
| fillna | description и отсутствующий enterprise_count |
| apply | нормализация пробелов в title |
| vectorized strings | объединение title и description в text |
| filtering / loc | enterprise subset для отчёта |
| groupby / agg | число обращений и среднее resolution time по category |
| merge | присоединение enterprise counts |
| sort_values / iloc | получить category с наибольшим средним временем |
| value_counts | распределение priority |
| to_numpy | test_schemas_and_arrays: перевод resolution_hours в ndarray |

`train.py` использует stratified train_test_split. Pipeline включает TfidfVectorizer и LogisticRegression. `fit` pipeline вызывает preprocessing fit_transform на train; при inference `predict_proba` выполняет только transform уже обученным vectorizer. В тесте estimator API виден отдельно, без разрушения production pipeline.

## NumPy и shapes

`generate_dataset`: random generator, np.resize, np.arange, choice, булевы маски и векторное умножение resolution hours. `features.row_softmax`: `[B,C] - [B,1]`, exp и нормализация по classes. `top_predictions`: argmax по строкам и advanced indexing `probabilities[np.arange(B), indices]`.

Utility softmax используется для чтения и теста численной устойчивости; sklearn сам считает вероятности. Torch inference переводит CPU Tensor в NumPy. Не подменяйте это дополнительным softmax над уже нормированными probabilities.

## Training и artifact

```bash
uv run python scripts/generate_dataset.py
uv run python scripts/train_sklearn.py
```

Metrics рядом с моделью: accuracy, macro precision, macro recall, macro F1 и confusion matrix с явным порядком labels. Joblib сохраняет pipeline целиком: vocabulary/IDF, estimator и classes. Не загружайте joblib от недоверенного источника: pickle допускает выполнение кода. Путь к артефакту задаёт configuration, а не HTTP caller.

Вероятность выбранного класса называется `score`; это не измеренная частота правильных ответов. Calibration и out-of-distribution detection не реализованы. Неизвестные слова игнорируются TF-IDF; низкий score направляет graph на дополнительный context.

## Ограничения оценки

Разделение происходит по строкам, формулировки одного template встречаются в train/test. Это не preprocessing leakage (vectorizer видит только train), но метрика измеряет в основном распознавание известных шаблонов. Для реального benchmark понадобились бы human-labelled data, split по времени/клиенту или template group, отдельные validation/test и анализ distribution shift. Torch holdout используется для выбора checkpoint, поэтому его метрика — validation, не независимый final test.

## Questions to answer after reading this code

1. Где fit_transform, transform, fit, predict и predict_proba?
2. Что входит в обученный artifact кроме коэффициентов classifier?
3. Почему нельзя fit vectorizer на всём dataset до split?
4. Что означает axis=1 и зачем keepdims=True?
5. Чем broadcasting отличается от Python loop?
6. Почему macro F1 важнее одной accuracy при imbalance?
7. Почему softmax confidence не равна calibrated correctness probability?
8. Чем template overlap отличается от leakage preprocessing?

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    # Векторизатор входит в Pipeline, чтобы словарь и IDF обучались только на
    # train. Отдельный fit_transform на всём CSV до split дал бы утечку данных.
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=12000)),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

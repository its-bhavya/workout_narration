"""Classifier catalogue; learned preprocessing is fitted separately per source fold."""
from typing import Any

def _build_classifiers(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    return {
        "rf": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "gbt": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            subsample=0.8, random_state=seed,
        ),
        "hist_gbt": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=5,
            class_weight="balanced", random_state=seed, early_stopping=False,
        ),
        "lr": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            solver="saga", random_state=seed,
        ),
        "svm": SVC(
            kernel="rbf", class_weight="balanced", probability=True,
            random_state=seed,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=7, metric="euclidean", n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=False,
            random_state=seed,
        ),
    }

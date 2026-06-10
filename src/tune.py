from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.data_loader import get_train_test_data

X_train, X_test, y_train, y_test = (
    get_train_test_data()
)

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "class_weight": [None, "balanced"],
    "max_iter": [1000]
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest F1 Score:")
print(grid_search.best_score_)
import joblib
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge

from dataset import build_dataset
from settings import *


def train_randomforest(
    dataset: str = DATASET_PATH,
    save_path: str = MODEL_SAVE_PATH,
    n_estimators: int = 300,
    max_depth: int = None,
):
    X, y, grid = build_dataset(dataset, n_points=SAMPLE_POINTS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred, multioutput="raw_values")
    print("MAE: B, theta, phi:", mae)

    joblib.dump({"model": model, "grid": grid}, save_path)


def train_extratrees(
    dataset: str = DATASET_PATH,
    save_path: str = MODEL_SAVE_PATH,
    n_estimators: int = 400,
    max_depth: int | None = None,
    max_features: str | int | float = "sqrt",
):
    X, y, grid = build_dataset(dataset, n_points=SAMPLE_POINTS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred, multioutput="raw_values")
    print("MAE: B, theta, phi:", mae)
    joblib.dump({"model": model, "grid": grid}, save_path)


def train_gradient_boosting(
    dataset: str = DATASET_PATH,
    save_path: str = MODEL_SAVE_PATH,
    learning_rate: float = 0.05,
    n_estimators: int = 600,
    max_depth: int = 3,
):
    X, y, grid = build_dataset(dataset, n_points=SAMPLE_POINTS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    base = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=SEED,
    )

    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred, multioutput="raw_values")
    print("MAE: B, theta, phi:", mae)
    joblib.dump({"model": model, "grid": grid}, save_path)


def train_svr_rbf(
    dataset: str = DATASET_PATH,
    save_path: str = MODEL_SAVE_PATH,
    C: float = 10.0,
    gamma: str | float = "scale",
    epsilon: float = 0.05,
):
    X, y, grid = build_dataset(dataset, n_points=SAMPLE_POINTS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)),
        ]
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred, multioutput="raw_values")
    print("MAE: B, theta, phi:", mae)
    joblib.dump({"model": model, "grid": grid}, save_path)


def train_knn(
    dataset: str = DATASET_PATH,
    save_path: str = MODEL_SAVE_PATH,
    n_neighbors: int = 15,
    weights: str = "distance",  # "uniform" or "distance"
    p: int = 2,  # 2: Euclidean, 1: Manhattan
):
    X, y, grid = build_dataset(dataset, n_points=SAMPLE_POINTS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    base = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        n_jobs=-1,
    )
    model = Pipeline(
        [("scaler", StandardScaler()), ("knn", MultiOutputRegressor(base, n_jobs=-1))]
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred, multioutput="raw_values")
    print("MAE: B, theta, phi:", mae)
    joblib.dump({"model": model, "grid": grid}, save_path)


def train_mlp(
    dataset: str = DATASET_PATH,
    save_path: str = MODEL_SAVE_PATH,
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
    activation: str = "relu",  # "relu", "tanh", "logistic"
    alpha: float = 1e-4,  # L2
    learning_rate_init: float = 1e-3,
    max_iter: int = 500,
):
    X, y, grid = build_dataset(dataset, n_points=SAMPLE_POINTS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred, multioutput="raw_values")
    print("MAE: B, theta, phi:", mae)
    joblib.dump({"model": model, "grid": grid}, save_path)


def train_linear(
    dataset: str = DATASET_PATH,
    save_path: str = MODEL_SAVE_PATH,
    use_ridge: bool = False,
    alpha: float = 1.0,
):
    X, y, grid = build_dataset(dataset, n_points=SAMPLE_POINTS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    if use_ridge:
        base = Ridge(alpha=alpha, random_state=SEED)
    else:
        base = LinearRegression()

    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    mae = mean_absolute_error(yte, pred, multioutput="raw_values")
    print("Linear/Ridge MAE: B, theta, phi:", mae)

    joblib.dump({"model": model, "grid": grid}, save_path)


def train_bayesian(
    dataset: str = DATASET_PATH,
    save_path: str = MODEL_SAVE_PATH,
    max_iter: int = 300,
    alpha_init: float = 1.0,
    lambda_init: float = 1.0,
):
    X, y, grid = build_dataset(dataset, n_points=SAMPLE_POINTS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    base = BayesianRidge(
        max_iter=max_iter,
        alpha_init=alpha_init,
        lambda_init=lambda_init,
        compute_score=True,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    mae = mean_absolute_error(yte, pred, multioutput="raw_values")
    print("Bayesian Ridge MAE: B, theta, phi:", mae)

    joblib.dump({"model": model, "grid": grid}, save_path)


if __name__ == "__main__":
    train_randomforest()

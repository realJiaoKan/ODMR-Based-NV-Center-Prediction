from uuid import uuid4

from sklearn.ensemble import RandomForestRegressor

from Train.lib.evaluation import delta_vec_norm_evaluator_np
from Train.lib.train import run_sklearn_ml

from settings import RANDOM_SEED
from Datasets.Spectra import load_dataset

NUM_RUNS = 3

N_ESTIMATORS = 400
MAX_DEPTH = None

RESULT_PATH = "Train/Results/RF"


if __name__ == "__main__":
    train_ds, test_ds = load_dataset()
    X_train, y_train = train_ds[:][0].numpy(), train_ds[:][1].numpy()
    X_test, y_test = test_ds[:][0].numpy(), test_ds[:][1].numpy()

    for run_idx in range(NUM_RUNS):
        run_id = str(uuid4())[:8]
        print(f"=== RF Run {run_id} ===")
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_SEED + run_idx,
            n_jobs=-1,
        )
        train_eval, test_eval = run_sklearn_ml(
            model, X_train, y_train, X_test, y_test, delta_vec_norm_evaluator_np
        )

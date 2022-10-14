import optuna
from optuna.samplers import TPESampler
from .seed_ import seed_everything
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import random
import os
import numpy as np

seed = 7575
sampler = TPESampler(seed=seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


def start_search(X_train, X_val, y_train, y_val):
    seed_everything()

    def objective(trial):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

        params = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 500),
            "n_estimators": trial.suggest_int("n_estimators", 1000, 5000),
            "max_depth": trial.suggest_int('max_depth', 2, 16),
            'l2_leaf_reg': trial.suggest_int('max_depth', 2, 10),
            'min_data_in_leaf': trial.suggest_int('max_depth', 2, 100),
            'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
            "eval_metric": "AUC",
            "loss_function": "MultiClass",
            "learning_rate": trial.suggest_uniform('learning_rate', 0.00001, 0.99),
            "min_data_in_leaf": trial.suggest_int('min_data_in_leaf', 2, 300),
            "grow_policy": "Lossguide",
            "od_type": "Iter",
            "od_wait": 10,
            'rsm': 1,
            "task_type": "CPU",
            "verbose": 0,
            "random_state": 7575

        }
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model = CatBoostClassifier(**params)

        model.fit(train_pool, eval_set=val_pool)
        #     score = cv(train_pool, params, fold_count=5, shuffle=True, plot=False, verbose=25)
        y_val_pred = model.predict(val_pool)

        return recall_score(y_val, y_val_pred, average="macro")

    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(objective, n_trials=120, n_jobs=-1)
    params_cat = study.best_params

    return params_cat

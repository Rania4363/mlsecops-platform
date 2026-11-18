import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["tenure", "monthly_charges", "contract", "support_tickets"]


def make_data(n, seed, drift=0.0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = np.clip(rng.normal(65 + 25 * drift, 20, n), 15, 150)
    contract = rng.choice(
        [0, 1, 2], n,
        p=[0.55 + 0.25 * drift, 0.25 - 0.10 * drift, 0.20 - 0.15 * drift],
    )
    tickets = rng.poisson(1.5 + 2.0 * drift, n)
    logit = (-0.5 - 0.035 * tenure + 0.02 * (monthly - 65)
             + 0.45 * tickets - 0.9 * contract)
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly,
        "contract": contract,
        "support_tickets": tickets,
        "churn": churn,
    })


def main():
    n = int(os.getenv("N_SAMPLES", "5000"))
    seed = int(os.getenv("SEED", "42"))
    drift = float(os.getenv("DRIFT", "0.0"))

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("churn")

    df = make_data(n, seed, drift)
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["churn"], test_size=0.2, random_state=seed,
        stratify=df["churn"],
    )

    with mlflow.start_run():
        params = {"n_estimators": 200, "max_depth": 8, "random_state": seed}
        model = RandomForestClassifier(**params).fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        mlflow.log_params({**params, "n_samples": n, "drift": drift})
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        })

        x_train.assign(churn=y_train).to_csv("/tmp/reference.csv", index=False)
        mlflow.log_artifact("/tmp/reference.csv", artifact_path="data")
        mlflow.sklearn.log_model(model, name="model",
                                 registered_model_name="churn-model")
        print("entraînement terminé")


if __name__ == "__main__":
    main()

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "Train.csv",
    BASE_DIR / "rawdata.csv",
    BASE_DIR / ".ipynb_checkpoints" / "rawdata-checkpoint.csv",
]
NUM_COLS = ["household_size", "age_of_respondent"]
CAT_COLS = [
    "country",
    "year",
    "location_type",
    "cellphone_access",
    "gender_of_respondent",
    "relationship_with_head",
    "marital_status",
    "education_level",
    "job_type",
]


def load_data():
    for path in DATA_CANDIDATES:
        if path.exists():
            print(f"Loading data from {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(f"No training CSV found. Looked in: {DATA_CANDIDATES}")


def make_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )


def evaluate(name, model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    print(f"\n{name}")
    print("  accuracy:", round(accuracy_score(y, pred), 4))
    print("  roc_auc :", round(roc_auc_score(y, proba), 4))
    print("  report  :\n", classification_report(y, pred))
    print("  matrix  :\n", confusion_matrix(y, pred))


def main():
    df = load_data()

    y = (df["bank_account"] == "Yes").astype(int)
    X = df.drop(columns=["bank_account", "uniqueid"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train rows: {len(X_train)}, validation rows: {len(X_val)}")

    pre = make_preprocessor()
    X_train_prepared = pre.fit_transform(X_train)
    X_val_prepared = pre.transform(X_val)
    print(
        "Prepared shapes:", X_train_prepared.shape, X_val_prepared.shape,
        "columns match:", X_train_prepared.shape[1] == X_val_prepared.shape[1],
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_prepared, y_train)
    evaluate("logistic_regression", lr, X_val_prepared, y_val)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_prepared, y_train)
    evaluate("random_forest", rf, X_val_prepared, y_val)

    xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1)
    xgb.fit(X_train_prepared, y_train)
    evaluate("xgboost", xgb, X_val_prepared, y_val)

    pipe = Pipeline(
        [
            ("pre", make_preprocessor()),
            ("clf", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1)),
        ]
    )
    param_grid = {
        "clf__max_depth": [3, 5],
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("\nBest parameters:", grid.best_params_)

    best = grid.best_estimator_
    evaluate("tuned_xgboost", best, X_val, y_val)

    model_path = BASE_DIR / "model.bin"
    joblib.dump(best, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

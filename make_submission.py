from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TEST_CANDIDATES = [
    BASE_DIR / "data" / "Test.csv",
    BASE_DIR / "Test.csv",
    BASE_DIR / "data" / "test.csv",
]


def load_test():
    for path in TEST_CANDIDATES:
        if path.exists():
            print(f"Loading test data from {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(f"No test CSV found. Looked in: {TEST_CANDIDATES}")


def main():
    test = load_test()
    model = joblib.load(BASE_DIR / "model.bin")

    features = test.drop(columns=["uniqueid"], errors="ignore")
    proba = model.predict_proba(features)[:, 1]

    submission = pd.DataFrame(
        {
            "unique_id": test["uniqueid"] + " x " + test["country"],
            "bank_account": (proba >= 0.5).astype(int),
        }
    )
    output = BASE_DIR / "submission.csv"
    submission.to_csv(output, index=False)
    print(f"Saved {len(submission)} rows to {output}")


if __name__ == "__main__":
    main()

import joblib
import pandas as pd
from flask import Flask, jsonify, request

model = joblib.load("model.bin")
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    record = request.get_json()
    features = pd.DataFrame([record]).drop(columns=["uniqueid"], errors="ignore")
    proba = float(model.predict_proba(features)[0, 1])
    return jsonify(
        {"probability": round(proba, 4), "bank_account": int(proba >= 0.5)}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)

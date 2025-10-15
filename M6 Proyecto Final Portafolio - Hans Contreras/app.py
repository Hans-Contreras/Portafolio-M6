# app.py – API Flask para inferencia
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model_pipeline.joblib")
FEATURES = ["age","income","debt_ratio","delinquencies","credit_history_years","has_mortgage","has_dependents","employment_years"]

@app.route("/health", methods=["GET"])
def health():
    return {"status":"ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    instances = data if isinstance(data, list) else [data]
    X = [[inst.get(k, 0) for k in FEATURES] for inst in instances]
    probs = model.predict_proba(np.array(X))[:,1].tolist()
    preds = (np.array(probs) >= 0.5).astype(int).tolist()
    out = [{"probability": float(p), "prediction": int(y)} for p, y in zip(probs, preds)]
    return jsonify(out if isinstance(data, list) else out[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

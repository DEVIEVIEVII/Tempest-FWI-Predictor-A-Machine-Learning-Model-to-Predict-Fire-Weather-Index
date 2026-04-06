from pathlib import Path
import json
import pickle
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

with open(MODELS / "ridge.pkl", "rb") as f:
    model = pickle.load(f)
with open(MODELS / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
features = json.loads((MODELS / "features.json").read_text(encoding="utf-8"))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form.get(f, 0)) for f in features]
    input_df = pd.DataFrame([values], columns=features)
    pred = model.predict(scaler.transform(input_df))[0]
    return render_template("home.html", prediction=round(float(pred), 3))

if __name__ == "__main__":
    app.run(debug=True)

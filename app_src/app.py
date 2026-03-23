from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_PATH = BASE_DIR / "data" / "forestfires.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "ridge.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS = ["temp", "rh", "wind", "rain", "ffmc", "dmc", "isi"]

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))


def load_pickle(path: Path):
    with path.open("rb") as model_file:
        return pickle.load(model_file)


def save_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as model_file:
        pickle.dump(obj, model_file)


def train_model_from_dataset():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset file was not found at '{DATA_PATH}'."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURE_COLUMNS + ["area"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}."
        )

    features = data[FEATURE_COLUMNS].astype(float)
    target = np.log1p(data["area"].astype(float))

    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = Ridge(alpha=1.0)
    model.fit(x_train_scaled, y_train)

    save_pickle(MODEL_PATH, model)
    save_pickle(SCALER_PATH, scaler)
    return model, scaler


try:
    if MODEL_PATH.is_file() and SCALER_PATH.is_file():
        model = load_pickle(MODEL_PATH)
        scaler = load_pickle(SCALER_PATH)
    else:
        model, scaler = train_model_from_dataset()
    model_load_error = None
except Exception as exc:
    model = None
    scaler = None
    model_load_error = str(exc)


@app.route("/")
def index():
    return render_template("index.html", model_load_error=model_load_error)


@app.route("/predict", methods=["POST"])
def predict():
    if model_load_error:
        return render_template("home.html", error=model_load_error)

    try:
        features = np.array(
            [[float(request.form[name]) for name in FEATURE_COLUMNS]]
        )
    except (KeyError, ValueError):
        return render_template(
            "home.html",
            error="Please enter valid numeric values for all input fields.",
        )

    features_scaled = scaler.transform(features)
    predicted_log_area = float(model.predict(features_scaled)[0])
    prediction = max(0.0, float(np.expm1(predicted_log_area)))

    return render_template("home.html", prediction_text=round(prediction, 3))


if __name__ == "__main__":
    app.run(debug=True)

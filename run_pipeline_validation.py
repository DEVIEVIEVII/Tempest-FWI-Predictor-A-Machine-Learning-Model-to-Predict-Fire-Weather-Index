from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Algerian_forest_fires_dataset_UPDATE.csv"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
TEMPLATES_DIR = ROOT / "templates"

MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

raw_lines = DATA_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
records = []
current_region = None
for line in raw_lines:
    text = line.strip()
    if not text:
        continue
    if "Region Dataset" in text:
        current_region = text.replace("Region Dataset", "").strip()
        continue
    if text.lower().startswith("day,month,year"):
        continue
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 14:
        continue
    records.append(parts[:14] + [current_region])

columns = [
    "day", "month", "year", "Temperature", "RH", "Ws", "Rain",
    "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "Classes", "Region"
]

df = pd.DataFrame(records, columns=columns)
for c in ["day", "month", "year"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["day", "month", "year"]).copy()

numeric_cols = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

df["Classes"] = df["Classes"].astype(str).str.strip().str.lower()
df["Region"] = df["Region"].astype(str).str.strip()
df["Region_encoded"] = df["Region"].map({r: i for i, r in enumerate(sorted(df["Region"].unique()))})
df["Classes_encoded"] = df["Classes"].map({"not fire": 0, "fire": 1})

df.to_csv(ARTIFACTS_DIR / "cleaned_fwi_dataset.csv", index=False)

candidate_features = [
    "Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "day", "month", "year", "Region_encoded"
]

corr_target = df[candidate_features + ["FWI"]].corr()["FWI"].drop("FWI").abs().sort_values(ascending=False)
selected_features = corr_target.head(8).index.tolist()

X = df[selected_features].copy()
y = df["FWI"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

(MODELS_DIR / "features.json").write_text(json.dumps(selected_features, indent=2), encoding="utf-8")

search = GridSearchCV(
    Ridge(),
    {"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]},
    scoring="neg_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
)
search.fit(X_train_scaled, y_train)
model = search.best_estimator_

with open(MODELS_DIR / "ridge.pkl", "wb") as f:
    pickle.dump(model, f)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

metrics = pd.DataFrame([
    {
        "split": "train",
        "MAE": mean_absolute_error(y_train, y_train_pred),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "R2": r2_score(y_train, y_train_pred),
    },
    {
        "split": "test",
        "MAE": mean_absolute_error(y_test, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "R2": r2_score(y_test, y_test_pred),
    },
])

metrics.to_csv(ARTIFACTS_DIR / "model_metrics.csv", index=False)

app_py = '''from pathlib import Path
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

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form.get(f, "0")) for f in features]
    input_df = pd.DataFrame([values], columns=features)
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    return render_template("home.html", prediction=round(float(prediction), 3))

if __name__ == "__main__":
    app.run(debug=True)
'''

index_html = '''<!doctype html>
<html>
<head><meta charset="utf-8"><title>FWI Predictor</title></head>
<body>
  <h1>FWI Predictor</h1>
  <form action="/predict" method="post">
    {% for feature in features %}
      <label>{{ feature }}:</label>
      <input type="number" step="any" name="{{ feature }}" required><br><br>
    {% endfor %}
    <button type="submit">Predict FWI</button>
  </form>
</body>
</html>
'''

home_html = '''<!doctype html>
<html>
<head><meta charset="utf-8"><title>FWI Result</title></head>
<body>
  <h1>Predicted Fire Weather Index (FWI)</h1>
  <h2>{{ prediction }}</h2>
  <a href="/">Try another prediction</a>
</body>
</html>
'''

(ROOT / "app.py").write_text(app_py, encoding="utf-8")
(TEMPLATES_DIR / "index.html").write_text(index_html, encoding="utf-8")
(TEMPLATES_DIR / "home.html").write_text(home_html, encoding="utf-8")

print("Best alpha:", search.best_params_["alpha"])
print(metrics)
print("Artifacts generated successfully.")
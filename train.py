import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (NO HEADER)
df = pd.read_csv("data/fwi-new.csv", header=None, delimiter=",")

# Assign column names
df.columns = [
    "day","month","year","Temperature","RH","Ws","Rain",
    "FFMC","DMC","DC","ISI","BUI","FWI","Classes"
]

# Drop unnecessary columns (day, month, year, DC, BUI, Classes)
df = df.drop(columns=["day","month","year","DC","BUI","Classes"], errors='ignore')

# Convert all to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing values
df = df.dropna()

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Features and target
X = df.drop("FWI", axis=1)
y = df["FWI"]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression Model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Create models directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model & scaler
pickle.dump(model, open("model/ridge.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("Model & Scaler Saved Successfully!")
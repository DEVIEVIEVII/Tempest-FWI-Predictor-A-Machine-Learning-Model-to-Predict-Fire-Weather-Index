# ===============================
# Milestone 1 : FWI Dataset EDA
# ===============================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load dataset
df = pd.read_csv("dataset/Algerian_forest_fires_dataset_UPDATE.csv", skiprows=1)

# Step 3: Clean column names
df.columns = df.columns.str.strip()

# Step 4: Display first rows
print("\nFirst 5 rows of dataset:")
print(df.head())

# Step 5: Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Step 6: Dataset information
print("\nDataset Info:")
print(df.info())

# Step 7: Statistical summary
print("\nDataset Description:")
print(df.describe())

# Step 8: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 9: Convert numeric columns
numeric_columns = [
    "Temperature","RH","Ws","Rain",
    "FFMC","DMC","DC","ISI","BUI","FWI"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove NaN values
df = df.dropna()

print("\nDataset after cleaning:")
print(df.shape)

# Step 10: Histogram
df.hist(figsize=(12,10))
plt.suptitle("Feature Distribution (Histogram)")
plt.show()

# Step 11: Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# Step 12: Scatter Plot
sns.scatterplot(x="Temperature", y="FWI", data=df)
plt.title("Temperature vs FWI")
plt.show()

# Step 13: Boxplot (Outliers)
plt.figure(figsize=(12,8))
sns.boxplot(data=df.select_dtypes(include=np.number))
plt.title("Outlier Detection")
plt.show()

# Step 14: Encode Region
if "Region" in df.columns:
    df["Region"] = df["Region"].map({
        "Bejaia":0,
        "Sidi-Bel Abbes":1
    })

print("\nEncoded Dataset:")
print(df.head())

# Step 15: Save cleaned dataset
df.to_csv("dataset/cleaned_fwi_dataset.csv", index=False)
print("\nCleaned dataset saved successfully!")

# ===============================
# Milestone 2 & 3 : Model + Evaluation
# ===============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Features & Target
X = df[["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI"]]
y = df["FWI"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge Model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

print("\nModel training completed")

# Prediction
y_pred = model.predict(X_test)

# ===============================
# Evaluation
# ===============================

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI")
plt.title("Actual vs Predicted")
plt.show()

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ===============================
# Alpha Tuning
# ===============================

print("\nAlpha Tuning Results:")
for a in [0.1, 1, 10]:
    temp_model = Ridge(alpha=a)
    temp_model.fit(X_train, y_train)
    temp_pred = temp_model.predict(X_test)
    print("Alpha:", a, "R2:", r2_score(y_test, temp_pred))

# ===============================
# Save Model
# ===============================

pickle.dump(model, open("ridge.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel and scaler saved successfully!")
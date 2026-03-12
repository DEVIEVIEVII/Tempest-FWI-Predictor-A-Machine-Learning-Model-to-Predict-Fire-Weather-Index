
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("cleaned_forestfires.csv")

print("Dataset Preview")
print(df.head())


numeric_df = df.select_dtypes(include=['number'])

print("\nCorrelation with Target (area):")
print(numeric_df.corr()["area"].sort_values(ascending=False))


X = df.drop("area", axis=1)
y = df["area"]


X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nScaler saved successfully!")


alphas = [0.01, 0.1, 1, 10, 100]

best_alpha = None
best_score = -999

print("\nAlpha Tuning Results")

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    score = r2_score(y_test, y_pred)

    print("Alpha:", alpha, " Test R2:", score)

    if score > best_score:
        best_score = score
        best_alpha = alpha

print("\nBest Alpha:", best_alpha)


ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_scaled, y_train)


y_pred_train = ridge.predict(X_train_scaled)
y_pred_test = ridge.predict(X_test_scaled)


train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

mse = mean_squared_error(y_test, y_pred_test)

print("\nModel Performance")
print("Training R2:", train_r2)
print("Testing R2:", test_r2)
print("Mean Squared Error:", mse)


with open("ridge.pkl", "wb") as f:
    pickle.dump(ridge, f)

print("\nModel saved successfully as ridge.pkl")
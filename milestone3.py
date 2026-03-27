

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv("cleaned_dataset.csv")


print("Dataset Preview")
print(data.head())



data['month'] = data['month'].astype('category').cat.codes
data['day'] = data['day'].astype('category').cat.codes



X = data.iloc[:, :-1]   
y = data.iloc[:, -1]   

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)


model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

print("\nModel Training Completed")



y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])  



mae = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error (MAE):", mae)



rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error (RMSE):", rmse)



r2 = r2_score(y_test, y_pred)

print("R² Score:", r2)



plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")

plt.show()


alphas = [0.01, 0.1, 1, 10, 100]

print("\nAlpha Tuning Results")

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Alpha:", a, "RMSE:", rmse)

    
best_alpha = 1
best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train, y_train)
y_pred_final = best_model.predict(X_test)
final_mae = mean_absolute_error(y_test, y_pred_final)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_r2 = r2_score(y_test, y_pred_final)


print("\nFinal Model Performance")
print("Final MAE:", final_mae)
print("Final RMSE:", final_rmse)
print("Final R2 Score:", final_r2)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_final)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Final Model: Actual vs Predicted")
plt.show() 
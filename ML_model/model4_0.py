import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------------------──
# 1. LOAD DATA
# ------------------------------------------------------------------------------──
data = pd.read_csv("cleaned_data.csv")
print(f"Dataset size: {len(data)} rows")


FEATURES = ['AQI', 'AQI_prev1', 'AQI_prev2', 
            'delta_AQI', 'delta2_AQI',
            'Temp', 'Humidity']

TARGET = 'PredictedAQI'

X = data[FEATURES]
y = data[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("--- Model Performance ---")
print(f"MAE  (avg error):       {mae:.2f} AQI units")
print(f"RMSE (penalizes spikes): {rmse:.2f} AQI units")
print(f"R²   (fit quality):      {r2:.4f}  (1.0 = perfect)")

# ------------------------------------------------------------------------------──
 # COMPARE OLD vs NEW MODEL
# ------------------------------------------------------------------------------──
old_model = LinearRegression()
old_model.fit(
    X_train[['AQI', 'AQI_prev1', 'AQI_prev2']],
    y_train
)
y_pred_old = old_model.predict(X_test[['AQI', 'AQI_prev1', 'AQI_prev2']])

mae_old  = mean_absolute_error(y_test, y_pred_old)
rmse_old = np.sqrt(mean_squared_error(y_test, y_pred_old))
r2_old   = r2_score(y_test, y_pred_old)

print("\n------ Old Model (AQI lags only) ------")
print(f"MAE:  {mae_old:.2f} | RMSE: {rmse_old:.2f} | R²: {r2_old:.4f}")
print("\n------ New Model (all features) ------")
print(f"MAE:  {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
print(f"\nImprovement: MAE reduced by {mae_old - mae:.2f} AQI units")

# ------------------------------------------------------------------------------──
# 7. PRINT COEFFICIENTS FOR ARDUINO
# ------------------------------------------------------------------------------──
print("\n------ Coefficients (copy these into Arduino) ------")
print(f"Intercept: {model.intercept_:.5f}")
for feature, coef in zip(FEATURES, model.coef_):
    print(f"  {feature:12s}: {coef:.5f}")

print("\n--- Arduino predictedAQI formula ------")
terms = []
for feature, coef in zip(FEATURES, model.coef_):
    sign = "+" if coef >= 0 else "-"
    terms.append(f"{abs(coef):.5f}f * {feature}")

intercept_sign = "+" if model.intercept_ >= 0 else "-"
print("predictedAQI = (int16_t)(")
for t in terms:
    print(f"    + {t}")
print(f"    {intercept_sign} {abs(model.intercept_):.5f}f")
print(");")

# ------------------------------------------------------------------------------──
# 8. EXAMPLE PREDICTION
# ------------------------------------------------------------------------------──
print("\n--- Example Prediction---")
sample = pd.DataFrame([[120, 110, 100, 10, 0, 28.0, 60.0]],
                      columns=FEATURES)
print(f"Input: {dict(zip(FEATURES, sample.values[0]))}")
print(f"Predicted AQI: {model.predict(sample)[0]:.1f}")

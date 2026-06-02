import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------------------------──
# LOAD & PREPARE
# ------------------------------------------------------------------------------──
data = pd.read_csv("cleaned_data.csv")

FEATURES = ['AQI', 'AQI_prev1', 'AQI_prev2',
            'delta_AQI', 'delta2_AQI',
            'Temp', 'Humidity']

X = data[FEATURES]
y = data['PredictedAQI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

# ------------------------------------------------------------------------------──
# PLOT 1: Actual vs Predicted over time
# ------------------------------------------------------------------------------──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('AQI Model v4.0 — Analysis', fontsize=14, fontweight='bold')

ax1 = axes[0, 0]
ax1.plot(data.index, data['AQI'], label='Actual AQI', color='steelblue', linewidth=1.2)
ax1.plot(data.index, data['PredictedAQI'], label='Predicted AQI', 
         color='orange', linewidth=1.2, alpha=0.8)
ax1.set_title('Actual vs Predicted AQI (full dataset)')
ax1.set_xlabel('Sample')
ax1.set_ylabel('AQI')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ------------------------------------------------------------------------------──
# PLOT 2: Scatter — Actual vs Predicted
# ------------------------------------------------------------------------------──
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred, alpha=0.5, color='steelblue', s=15)
# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect fit')
ax2.set_title(f'Actual vs Predicted (test set)\nMAE={mae:.1f}, R²={r2:.4f}')
ax2.set_xlabel('Actual AQI')
ax2.set_ylabel('Predicted AQI')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ------------------------------------------------------------------------------──
# PLOT 3: Residuals (error distribution)
# ------------------------------------------------------------------------------──
ax3 = axes[1, 0]
residuals = y_test.values - y_pred
ax3.hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax3.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax3.set_title('Residuals Distribution\n(should be centered at 0)')
ax3.set_xlabel('Error (Actual − Predicted)')
ax3.set_ylabel('Count')
ax3.grid(True, alpha=0.3)

# ------------------------------------------------------------------------------──
# PLOT 4: Feature importance (coefficients)
# ------------------------------------------------------------------------------──
ax4 = axes[1, 1]
coefs = pd.Series(model.coef_, index=FEATURES).sort_values()
colors = ['tomato' if c < 0 else 'steelblue' for c in coefs]
coefs.plot(kind='barh', ax=ax4, color=colors)
ax4.axvline(0, color='black', linewidth=0.8)
ax4.set_title('Feature Coefficients\n(magnitude = influence on prediction)')
ax4.set_xlabel('Coefficient value')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------──
# PLOT 5: Raw vs Corrected Gas (sanity check)
# ------------------------------------------------------------------------------──
if 'RawGas' in data.columns and 'CorrectedGas' in data.columns:
    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.index, data['RawGas'],       label='Raw Gas',       color='tomato',    linewidth=1)
    ax.plot(data.index, data['CorrectedGas'], label='Corrected Gas', color='steelblue', linewidth=1)
    ax.set_title('Raw vs Corrected Gas Reading (correction effect)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Sensor Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('correction_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    diff = data['RawGas'] - data['CorrectedGas']
    print(f"Average correction applied: {diff.mean():.1f} units")
    print(f"Max correction applied:     {diff.max():.1f} units")

print("Graphs saved.")

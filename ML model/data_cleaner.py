import pandas as pd

# Load dataset
data = pd.read_csv("data.csv")

print("Original shape:", data.shape)
print("Columns found:", list(data.columns))

# -----------------------------
# 1. Remove duplicate headers
# -----------------------------
data = data[data['Time'] != 'Time']

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop NaN rows
data = data.dropna()

# -----------------------------
# 2. Remove useless zero regions
# -----------------------------
data = data[(data['AQI'] != 0) | (data['PredictedAQI'] != 0)]

# -----------------------------
# 3. Remove very small noise values
# -----------------------------
data = data[data['AQI'] > 5]

# -----------------------------
# 4. Remove saturation (ceiling effect)
# -----------------------------
data = data[data['PredictedAQI'] < 500]
data = data[data['AQI'] < 500]

# -----------------------------
# 5. Remove physically impossible sensor values
# -----------------------------
# Temperature: MQ135 + DHT11 typical indoor/outdoor range
data = data[(data['Temp'] >= 0) & (data['Temp'] <= 60)]

# Humidity: 0–100% is the only valid range
data = data[(data['Humidity'] >= 0) & (data['Humidity'] <= 100)]

# CorrectedGas should be positive
data = data[data['CorrectedGas'] > 0]

# -----------------------------
# 6. Engineer new features
# -----------------------------

# Lag features (time series memory)
data['AQI_prev1'] = data['AQI'].shift(1)
data['AQI_prev2'] = data['AQI'].shift(2)

# Rate of change — how fast AQI is moving (velocity)
data['delta_AQI'] = data['AQI'] - data['AQI_prev1']

# Acceleration — is it speeding up or slowing down?
data['delta2_AQI'] = data['delta_AQI'] - data['delta_AQI'].shift(1)

# Drop rows with NaN from lag/delta operations
data = data.dropna()

# -----------------------------
# 7. Reset index
# -----------------------------
data = data.reset_index(drop=True)

print("Cleaned shape:", data.shape)
print("\nFeature summary:")
print(data[['Temp', 'Humidity', 'CorrectedGas', 'AQI', 
            'AQI_prev1', 'AQI_prev2', 'delta_AQI', 'delta2_AQI',
            'PredictedAQI']].describe())

# -----------------------------
# 8. Save cleaned dataset
# -----------------------------
data.to_csv("cleaned_data.csv", index=False)
print("\nCleaned dataset saved as cleaned_data.csv")

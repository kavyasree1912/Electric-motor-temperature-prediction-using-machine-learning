import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv('measures_v2.csv')
data.dropna(inplace=True)

# Define features and target
features = ['ambient', 'torque', 'coolant', 'u_d', 'u_q', 'motor_speed', 'i_d', 'i_q']
X = data[features]
y = data['pm']  # Motor temperature

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler (🔁 both must match)
joblib.dump(model, 'sensor_model.pkl')
joblib.dump(scaler, 'sensor_scaler.pkl')

# Optional evaluation
score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score:.2f}")
print("Sensor model + scaler trained and saved ✅")

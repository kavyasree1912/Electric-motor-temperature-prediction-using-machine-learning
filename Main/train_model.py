import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
data = pd.read_csv("measures_v2.csv")

# Select input features and target
X = data.drop("stator_winding", axis=1)
y = data["stator_winding"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=10, max_depth=5),
   # "SVM": SVR()
}

best_model = None
best_score = float('inf')

for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} MSE: {mse:.2f}")

        if mse < best_score:
            best_score = mse
            best_model = model
    except Exception as e:
        print(f"{name} failed: {e}")

# Save best model
if best_model:
    pickle.dump(best_model, open("best_model.pkl", "wb"))
    print("✅ Best model saved as best_model.pkl")
else:
    print("❌ No model trained successfully.")

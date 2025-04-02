# retrain_scaler.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler  # or your original scaler type

# Load your training dataset (replace with your actual data)
# Example data:
data = pd.DataFrame({
    "savings_rate": [0.2, 0.3, 0.4, 0.15, 0.25]  # 1 feature
})

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(data[["savings_rate"]])  # Fit on the 1 feature

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler retrained and saved!")
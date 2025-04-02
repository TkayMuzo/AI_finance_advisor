# retrain_model.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Example training data (1 feature)
X_train = pd.DataFrame({
    "savings_rate": [0.2, 0.3, 0.4, 0.15, 0.25]
})
y_train = [0.75, 0.80, 0.90, 0.60, 0.70]  # Example target values

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "regression_model.pkl")
print("Model retrained and saved!")
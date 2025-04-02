import pandas as pd
import joblib
from sklearn.linear_model import Ridge

def train_model():
    df = pd.read_csv('financial_features.csv')
    features = ['Total_Income_Lag_1', 'Total_Expenses_Lag_1', 'Savings_Rate']
    target = 'Total_Expenses_Lag_1'  # Example target
    
    model = Ridge(alpha=0.1)
    model.fit(df[features], df[target])
    
    joblib.dump(model, 'model.pkl')
    print(f"Trained on features: {features}")

if __name__ == '__main__':
    train_model()
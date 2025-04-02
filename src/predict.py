import pandas as pd
import joblib
from feature_engineering import engineer_features

class Predictor:
    def __init__(self):
        self.model = joblib.load('model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.features = ['Total_Income_Lag_1', 'Total_Expenses_Lag_1', 'Savings_Rate']
    
    def predict(self, input_path):
        raw_data = pd.read_csv(input_path)
        processed = engineer_features(raw_data, mode='inference')
        
        # Scale features
        scaled = self.scaler.transform(processed[self.features])
        
        # Predict
        processed['Prediction'] = self.model.predict(scaled)
        
        # Format output
        return processed.rename(columns={'Date': 'Month'})[['Month', 'Prediction']]

if __name__ == '__main__':
    advisor = Predictor()
    results = advisor.predict('monthly_summary.csv')
    print(results.to_markdown(index=False))
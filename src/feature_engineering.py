# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Common feature configuration
FEATURE_COLS = ['Total_Income_Lag_1', 'Total_Expenses_Lag_1', 'Savings_Rate']

def load_data(filepath: str = 'processed_transactions.csv') -> pd.DataFrame:
    """Load input data with error handling"""
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df = df.drop(columns=['Category'], errors='ignore')
        df['Amount'] = df['Amount'].astype(float)
        logging.info(f"Loaded {len(df)} records from {filepath}")
        return df.sort_values('Date')
    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        sys.exit(1)

def engineer_features(df: pd.DataFrame, mode: str = 'train') -> pd.DataFrame:
    """Feature engineering pipeline with strict feature alignment"""
    try:
        if mode == 'train':
            # Training data processing
            monthly = df.groupby(pd.Grouper(key='Date', freq='M')).agg(
                Total_Income=('Amount', lambda x: x[x > 0].sum()),
                Total_Expenses=('Amount', lambda x: abs(x[x < 0].sum()))
            ).reset_index()

            # Generate temporal features
            monthly['Total_Income_Lag_1'] = monthly['Total_Income'].shift(1)
            monthly['Total_Expenses_Lag_1'] = monthly['Total_Expenses'].shift(1)
            monthly = monthly.dropna()
            
            # Calculate financial ratios
            monthly['Savings_Rate'] = ((monthly['Total_Income'] - monthly['Total_Expenses']) 
                                      / monthly['Total_Income'])
            
            # Validate features
            missing = set(FEATURE_COLS) - set(monthly.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")

            # Scale features
            scaler = StandardScaler()
            monthly[FEATURE_COLS] = scaler.fit_transform(monthly[FEATURE_COLS])
            dump(scaler, 'scaler.pkl')
            
            return monthly[FEATURE_COLS]
        
        elif mode == 'inference':
            # Inference data processing
            df = df.rename(columns={
                'total_income': 'Total_Income',
                'total_expense': 'Total_Expenses',
                'Month': 'Date'
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Generate identical features
            df['Total_Income_Lag_1'] = df['Total_Income'].shift(1)
            df['Total_Expenses_Lag_1'] = df['Total_Expenses'].shift(1)
            df = df.dropna()
            df['Savings_Rate'] = ((df['Total_Income'] - df['Total_Expenses']) 
                                 / df['Total_Income'])
            
            # Validate and return
            missing = set(FEATURE_COLS) - set(df.columns)
            if missing:
                raise ValueError(f"Missing inference features: {missing}")
                
            return df[FEATURE_COLS]
        
        else:
            raise ValueError("Invalid mode. Use 'train' or 'inference'")
    
    except Exception as e:
        logging.error(f"Feature engineering failed: {str(e)}")
        sys.exit(1)

def main():
    """Main workflow with enhanced validation"""
    try:
        df = load_data()
        engineered_df = engineer_features(df, mode='train')
        engineered_df.to_csv('financial_features.csv', index=False)
        logging.info("Successfully generated features with columns: %s", engineered_df.columns.tolist())
    
    except Exception as e:
        logging.error("Feature engineering pipeline failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
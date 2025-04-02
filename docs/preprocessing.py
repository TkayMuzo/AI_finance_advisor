import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_transactions(input_path, output_path):
    # Load raw data
    df = pd.read_csv(input_path)
    
    # Clean redundant columns
    df = df.drop(columns=['Description', 'Category'])  # Remove duplicates
    
    # Convert dates and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Handle outliers in transaction amounts (cap at 95th percentile)
    q95 = df['Amount'].abs().quantile(0.95)
    df['Amount'] = np.where(df['Amount'].abs() > q95, 
                           np.sign(df['Amount']) * q95, 
                           df['Amount'])
    
    # Aggregate to monthly level
    monthly = df.groupby('Month').agg(
        Income=pd.NamedAgg(column='Amount', aggfunc=lambda x: x[x > 0].sum()),
        Expenses=pd.NamedAgg(column='Amount', aggfunc=lambda x: abs(x[x < 0].sum())),
        Num_Transactions=pd.NamedAgg(column='Amount', aggfunc='count'),
        Avg_Transaction=pd.NamedAgg(column='Amount', aggfunc='mean')
    ).reset_index()
    
    # Calculate financial metrics
    monthly['Net_Savings'] = monthly['Income'] - monthly['Expenses']
    monthly['Savings_Rate'] = monthly['Net_Savings'] / monthly['Income']
    
    # Create time-based features
    monthly['Month'] = pd.to_datetime(monthly['Month'])
    monthly = monthly.sort_values('Month')
    
    # Lag features (3-month history)
    for lag in [1, 2, 3]:
        monthly[f'Income_Lag_{lag}'] = monthly['Income'].shift(lag)
        monthly[f'Expenses_Lag_{lag}'] = monthly['Expenses'].shift(lag)
    
    # Rolling averages
    monthly['Income_3MMA'] = monthly['Income'].rolling(3).mean()
    monthly['Expenses_3MMA'] = monthly['Expenses'].rolling(3).mean()
    
    # Remove incomplete months
    monthly = monthly.dropna()
    
    # Format final output
    features = ['Month', 'Income', 'Expenses', 'Net_Savings', 'Savings_Rate',
                'Num_Transactions', 'Avg_Transaction', 'Income_Lag_1',
                'Expenses_Lag_1', 'Income_3MMA', 'Expenses_3MMA']
    
    # Scale numeric features
    scaler = StandardScaler()
    scaled_cols = ['Income', 'Expenses', 'Net_Savings', 'Num_Transactions', 
                   'Avg_Transaction']
    monthly[scaled_cols] = scaler.fit_transform(monthly[scaled_cols])
    
    # Save processed data
    monthly[features].to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_transactions(
        input_path='processed_transactions.csv',
        output_path='monthly_financials.csv'
    )
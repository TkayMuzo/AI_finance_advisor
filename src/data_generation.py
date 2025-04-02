from faker import Faker
import pandas as pd
import random

fake = Faker()
categories = ['Groceries', 'Rent', 'Utilities', 'Dining', 'Entertainment', 'Transportation', 'Healthcare', 'Salary']

def generate_transaction():
    # Randomly choose debit or credit. For income, we'll use credit.
    transaction_type = random.choice(['debit', 'credit'])
    
    # Set amount ranges depending on the type
    if transaction_type == 'credit':
        amount = round(random.uniform(1000, 5000), 2)  # For income
    else:
        amount = round(random.uniform(5, 500), 2)      # For expenses

    transaction = {
        'Date': fake.date_between(start_date='-1y', end_date='today'),
        'Description': fake.sentence(nb_words=6),
        'Amount': amount,
        'Type': transaction_type,
        'Category': random.choice(categories) if transaction_type == 'debit' else 'Income'
    }
    return transaction

num_transactions = 1000  # or any number you need

transactions = [generate_transaction() for _ in range(num_transactions)]

df = pd.DataFrame(transactions)

print(df.head())
print(df.describe())

df.to_csv('synthetic_transactions.csv', index=False)

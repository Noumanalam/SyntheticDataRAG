import pandas as pd

data = pd.read_csv('data.csv', encoding='ISO-8859-1')

print("First 5 rows:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

data = data.dropna(subset=['CustomerID'])

data = data.drop(columns=['Description'], errors='ignore')

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

data.to_csv('cleaned_data.csv', index=False)
print("Cleaned dataset saved as 'cleaned_data.csv'")
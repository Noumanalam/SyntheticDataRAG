import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

real_data = pd.read_csv('cleaned_data.csv').head(100)

synthetic_data = pd.DataFrame([
    {"CustomerID": 12345, "Quantity": 5, "UnitPrice": 10.5, "Country": "France"},
    {"CustomerID": 54321, "Quantity": 3, "UnitPrice": 8.75, "Country": "Germany"}
])  


plt.figure(figsize=(10, 6))
sns.histplot(real_data['Quantity'], color='blue', label='Real', stat='density')
sns.histplot(synthetic_data['Quantity'], color='orange', label='Synthetic', stat='density')
plt.legend()
plt.title('Quantity Distribution: Real vs Synthetic')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(real_data['UnitPrice'], color='blue', label='Real', stat='density')
sns.histplot(synthetic_data['UnitPrice'], color='orange', label='Synthetic', stat='density')
plt.legend()
plt.title('UnitPrice Distribution: Real vs Synthetic')
plt.show()

ks_quantity = ks_2samp(real_data['Quantity'], synthetic_data['Quantity'])
ks_unitprice = ks_2samp(real_data['UnitPrice'], synthetic_data['UnitPrice'])
print(f"KS Test (Quantity): statistic={ks_quantity.statistic}, p-value={ks_quantity.pvalue}")
print(f"KS Test (UnitPrice): statistic={ks_unitprice.statistic}, p-value={ks_unitprice.pvalue}")


real_data['HighValue'] = (real_data['UnitPrice'] > real_data['UnitPrice'].median()).astype(int)
synthetic_data['HighValue'] = (synthetic_data['UnitPrice'] > synthetic_data['UnitPrice'].median()).astype(int)

features = ['Quantity', 'UnitPrice']
X_real = real_data[features]
y_real = real_data['HighValue']
X_synth = synthetic_data[features]
y_synth = synthetic_data['HighValue']

X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.3, random_state=42)

rf_real = RandomForestClassifier(random_state=42)
rf_real.fit(X_train, y_train)
y_pred_real = rf_real.predict(X_test)
real_accuracy = accuracy_score(y_test, y_pred_real)
print(f"Accuracy (Real Data): {real_accuracy}")

if len(X_synth) > 1:  
    X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(X_synth, y_synth, test_size=0.3, random_state=42)
    rf_synth = RandomForestClassifier(random_state=42)
    rf_synth.fit(X_synth_train, y_synth_train)
    y_synth_pred = rf_synth.predict(X_synth_test)
    synth_accuracy = accuracy_score(y_synth_test, y_synth_pred)
    print(f"Accuracy (Synthetic Data): {synth_accuracy}")
else:
    print("Not enough synthetic data for ML evaluation.")





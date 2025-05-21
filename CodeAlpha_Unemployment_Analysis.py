# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# LOAD DATA
df1 = pd.read_csv(r"C:\Users\alphi\Downloads\unemployment\Unemployment in India.csv")
df2 = pd.read_csv(r"C:\Users\alphi\Downloads\unemployment\Unemployment_Rate_upto_11_2020.csv")

# CLEAN DATA
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')
df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')

# 1. STATE-WISE UNEMPLOYMENT TRENDS
plt.figure(figsize=(14, 8))
statewise = df2.groupby(['Region', 'Date'])['Estimated Unemployment Rate (%)'].mean().reset_index()
sns.lineplot(data=statewise, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('State-wise Unemployment Rate Trends (2020)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend(title='States', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. URBAN VS RURAL UNEMPLOYMENT
plt.figure(figsize=(10, 6))
urban_rural = df1.groupby(['Area', 'Date'])['Estimated Unemployment Rate (%)'].mean().reset_index()
sns.lineplot(data=urban_rural, x='Date', y='Estimated Unemployment Rate (%)', hue='Area')
plt.title('Urban vs Rural Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. PREDICTIVE MODELING (LINEAR REGRESSION using df1)
df1_model = df1[['Date', 'Estimated Unemployment Rate (%)']].dropna()
df1_model = df1_model.sort_values('Date')

X = df1_model['Date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = df1_model['Estimated Unemployment Rate (%)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

dates_test = [pd.Timestamp.fromordinal(int(x[0])) for x in X_test]

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(dates_test, y_test, label='Actual', marker='o')
plt.plot(dates_test, y_pred, label='Predicted', linestyle='--', color='red')
plt.title(f'Linear Regression Prediction (RMSE = {rmse:.2f})')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

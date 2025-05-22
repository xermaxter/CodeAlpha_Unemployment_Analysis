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

df1.dropna(subset=['Date'], inplace=True)
df2.dropna(subset=['Date'], inplace=True)

# 1. STATE-WISE UNEMPLOYMENT TRENDS
plt.figure(figsize=(14, 8))
statewise = df2.groupby(['Region', 'Date'])['Estimated Unemployment Rate (%)'].mean().reset_index()
sns.lineplot(data=statewise, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('State-wise Unemployment Rate Trends (2020)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend(title='States', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. URBAN VS RURAL UNEMPLOYMENT TRENDS
plt.figure(figsize=(10, 6))
urban_rural = df1.groupby(['Area', 'Date'])['Estimated Unemployment Rate (%)'].mean().reset_index()
sns.lineplot(data=urban_rural, x='Date', y='Estimated Unemployment Rate (%)', hue='Area')
plt.title('Urban vs Rural Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. COVID-19 IMPACT ANALYSIS (March 2020 onwards)
covid_df = df2[df2['Date'] >= '2020-03-01']
pre_covid_df = df2[df2['Date'] < '2020-03-01']

avg_pre_covid = pre_covid_df['Estimated Unemployment Rate (%)'].mean()
avg_during_covid = covid_df['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(6, 4))
sns.barplot(x=['Pre-COVID', 'During COVID'], y=[avg_pre_covid, avg_during_covid], palette='coolwarm')
plt.title('Average Unemployment Rate Before and During COVID-19')
plt.ylabel('Unemployment Rate (%)')
plt.tight_layout()
plt.show()

# 4. SEASONAL TRENDS (MONTHLY AVERAGE)
df2['Month'] = df2['Date'].dt.month_name()
monthly_avg = df2.groupby('Month')['Estimated Unemployment Rate (%)'].mean()
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, order=month_order, palette='viridis')
plt.title('Average Unemployment Rate by Month (Seasonal Trend)')
plt.xlabel('Month')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. LINEAR REGRESSION PREDICTION
df1_model = df1[['Date', 'Estimated Unemployment Rate (%)']].dropna().sort_values('Date')
X = df1_model['Date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = df1_model['Estimated Unemployment Rate (%)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

dates_test = [pd.Timestamp.fromordinal(int(x[0])) for x in X_test]
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

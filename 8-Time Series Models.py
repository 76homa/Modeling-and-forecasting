import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split  # Add this import

# Read data using pandas
excel_file_path = 'C:/Users/homa.behmardi/Downloads/Sirjan.xlsx'
sheet_name = 'Sheet1'
data = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# Replace X and y with your actual feature and target variable data
X = data[['thr', 'bw', 'PRB']]
y = data['payload']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check stationarity using ADF test
result = adfuller(y)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print('Critical Values:')
for key, value in result[4].items():
    print(f'  {key}: {value}')

# If not stationary, apply differencing
d = 0  # Initialize differencing order
if result[1] > 0.05:  # If p-value > 0.05, differencing is needed
    d = 1  # You can try higher orders if needed

# Plot ACF and PACF to determine p and q values
plot_acf(y.diff(periods=d).dropna(), lags=30)
plot_pacf(y.diff(periods=d).dropna(), lags=30)
plt.show()

# Define and fit the ARIMA model
p, q = 2, 2  # Adjust these values based on ACF and PACF plots
model = ARIMA(y, order=(p, d, q))
results = model.fit()

# Make predictions
y_pred = results.predict(start=len(y), end=len(y) + len(X_test) - 1, typ='levels', dynamic=False)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Display the evaluation results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r_squared}")

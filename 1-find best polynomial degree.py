import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Read data using pandas
excel_file_path = 'C:/Users/homa.behmardi/Downloads/Sirjan.xlsx'
sheet_name = 'Sheet1'  # Replace with the correct sheet name
data = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# Replace X and y with your actual feature and target variable data
X = data[['thr', 'bw', 'PRB']]
y = data['payload']  # Replace with your target variable (e.g., y = data['Payload'])

# Split your data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a range of polynomial degrees to test (e.g., 1 to 20)
degrees = np.arange(2, 8)
mse_results = []
r_squared_results = []

for degree in degrees:
    # Create polynomial features for the given degree
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train a linear regression model on the polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_poly)
    
    # Calculate Mean Squared Error (MSE) for this degree
    mse = mean_squared_error(y_test, y_pred)
    mse_results.append(mse)
    
    # Calculate R-squared for this degree
    r_squared = r2_score(y_test, y_pred)
    r_squared_results.append(r_squared)

# Plot the MSE and R-squared results for different degrees
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(degrees, mse_results, marker='o')
plt.title('Mean Squared Error (MSE) for Different Polynomial Degrees')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(degrees, r_squared_results, marker='o', color='orange')
plt.title('R-squared for Different Polynomial Degrees')
plt.xlabel('Polynomial Degree')
plt.ylabel('R-squared')
plt.grid(True)

plt.tight_layout()
plt.show()

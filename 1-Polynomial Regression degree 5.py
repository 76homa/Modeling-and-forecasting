import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Read data using pandas
excel_file_path = 'C:/Users/homa.behmardi/Downloads/Sirjan.xlsx'
sheet_name = 'Sheet1'  # Replace with the correct sheet name
data = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# Replace X and y with your actual feature and target variable data
X = data[['thr', 'bw', 'PRB']]
y = data['payload']  # Replace with your target variable (e.g., y = data['Payload'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features with degree 2
poly_features = PolynomialFeatures(degree=5)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Train a linear regression model with polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Display the evaluation results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r_squared}")

# Get the coefficients
intercept = model.intercept_
coefficients = model.coef_

# Get the feature names (column names of X_train_poly)
feature_names = poly_features.get_feature_names(input_features=X.columns)

# Create the nonlinear regression formula (Degree 2) with variable names
linear_formula = f"Y = {intercept:.4f} "
for i, coef in enumerate(coefficients[1:]):
    linear_formula += f"+ {coef:.4f} * {feature_names[i + 1]}"

print("Nonlinear Regression Formula (Degree 2):")
print(linear_formula)

# Create a scatter plot with the regression line
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred, alpha=0.6)

# Add a regression line
plt.plot(np.arange(min(y_test), max(y_test)), np.arange(min(y_test), max(y_test)), color='red', linestyle='--')

# Add labels and a title
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Scatter Plot with Regression Line')

# Show the plot
plt.show()

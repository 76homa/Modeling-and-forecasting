import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read data using pandas
excel_file_path = 'C:/Users/homa.behmardi/Downloads/Sirjan.xlsx'
sheet_name = 'Sheet1'  # Replace with the correct sheet name
data = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# Replace X and y with your actual feature and target variable data
X = data[['thr', 'bw', 'PRB']]
y = data['payload']  # Replace with your target variable (e.g., y = data['Payload'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Display the evaluation results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r_squared}")

# Optionally, you can save the trained model for future use
# from joblib import dump
# dump(model, 'linear_regression_model.joblib')

# Get the coefficients
coefficients = model.coef_

# Get the intercept
intercept = model.intercept_

# Create the linear formula
linear_formula = "Y = {:.4f}".format(intercept)
for i, coef in enumerate(coefficients):
    linear_formula += " + {:.4f} * X{}".format(coef, i + 1)
    
print(f"Intercept (β₀): {intercept}")    
print("Coefficients:")
for i, coef in enumerate(coefficients):
    feature_name = X.columns[i]  # Get the name of the corresponding feature
    print(f"  Coefficient for {feature_name}: {coef}")
print(f"Intercept: {intercept}")
linear_formula = f"Y = {intercept:.4f}"
for i, coef in enumerate(coefficients):
    feature_name = X.columns[i]  # Get the name of the corresponding feature
    linear_formula += f" + {coef:.4f} * {feature_name}"

print("Linear Formula:")
print(linear_formula)



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Create a DataFrame with actual and predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results, x='Actual', y='Predicted', alpha=0.6)

# Add the regression line to the scatter plot
sns.regplot(data=results, x='Actual', y='Predicted', scatter=False, color='red')

# Add labels and a title
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Scatter Plot with Linear Regression Fit')

# Show the plot
plt.show()

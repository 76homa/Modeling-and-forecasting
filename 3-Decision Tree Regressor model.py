import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Read the data
excel_file_path = 'C:/Users/homa.behmardi/Downloads/Sirjan.xlsx'
sheet_name = 'Sheet1'  # Replace with the correct sheet name
data = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# Replace X and y with your actual feature and target variable data
X = data[['thr', 'bw', 'PRB']]
y = data['payload']  # Replace with your target variable (e.g., y = data['Payload'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DecisionTreeRegressor model
model = DecisionTreeRegressor(random_state=42)  # Random state for reproducibility

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions for the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Print the evaluation results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r_squared}")

# Create a scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Scatter Plot of Actual vs. Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.show()

# Get feature importances from the RandomForestRegressor model
feature_importance = model.feature_importances_

# Visualize feature importances in a bar chart
import matplotlib.pyplot as plt

# Display feature importances as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance, tick_label=X.columns)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance in DecisionTreeRegressor')
plt.show()

import pandas as pd

# Create a DataFrame for the new data points you want to predict
# In this example, specify the values of independent variables for the new data
new_data = pd.DataFrame({
    'thr': [0, 0.1, 0],                                # New values for connecteduser
    'bw': [0.25, 0.19, 0.27],
    'PRB': [0.11, 0.11, 0.11]                                     # New values for PRB
})

# Predict Traffic for the new data points
predicted_traffic = model.predict(new_data)

# The predicted values for Traffic for the new data points
print(predicted_traffic)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Step 1: Load and prepare the data
data = pd.read_csv('airfoil_database.csv')

# Select features and target
features = data[['wu1', 'wu2', 'wu3', 'wl1', 'wl2', 'wl3']]
target = data['cl']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build the Neural Network model
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)

# Step 5: Train the model
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate performance
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')

cv_scores = cross_val_score(model, features, target, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-Validation MSE: {-cv_scores.mean()}')

# Step 7: Make predictions
print(f'Predictions: {y_pred}')

# Step 7: Visualization
plt.scatter(y_test, y_pred)
plt.xlabel('Actual $C_l$')
plt.ylabel('Predicted $C_l$')
plt.title('Actual vs Predicted $C_l$')
plt.plot([-1, 1], [-1, 1], color='red', linestyle='--')  # Line y=x for reference
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.show()

residuals = y_test - y_pred

plt.hist(residuals, bins=20)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Step 9: Making predictions
def predict_cl(wu1, wu2, wu3, wl1, wl2, wl3, model, scaler):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[wu1, wu2, wu3, wl1, wl2, wl3]], columns=['wu1', 'wu2', 'wu3', 'wl1', 'wl2', 'wl3'])
    
    # Scale the input data using the scaler
    input_scaled = scaler.transform(input_data)
    
    # Make the prediction
    cl_prediction = model.predict(input_scaled)
    
    return cl_prediction[0]

# Example usage
wu1 = 0.3
wu2 = 0.3
wu3 = 0.2
wl1 = -0.2
wl2 = 0.1
wl3 = 0.0

predicted_cl = predict_cl(wu1, wu2, wu3, wl1, wl2, wl3, model, scaler)
print(f'Predicted C_l: {predicted_cl}')

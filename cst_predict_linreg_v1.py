#%% IMPORTING LIBRARIES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#%% READING THE DATABASE AND PREPROCESS
# Read the CSV database file
df = pd.read_csv('airfoil_database.csv')

# Display the loaded database
print("\nDATABASE:")
print(df)

# Define features (geometric variables) and target variable (cl)
X = df[['wu1', 'wu2', 'wu3', 'wl1', 'wl2', 'wl3']]
y = df['cl']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% CREATING THE MODEL
# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')

#%% VISUALIZATION AND POSTPROCCES
# Preprocessing Visualization: scatter plots of cl vs m, p, t
plt.figure(figsize=(12, 8))
plt.suptitle('Pre/Post Process', fontsize=16)

# Plot cl vs m
plt.subplot(2, 3, 1)
sns.scatterplot(data=df, x='m', y='cl', color='blue')
plt.title('Lift Coefficient (cl) vs m')
plt.xlabel('m')
plt.ylabel('cl')
plt.grid()

# Plot cl vs p
plt.subplot(2, 3, 2)
sns.scatterplot(data=df, x='p', y='cl', color='green')
plt.title('Lift Coefficient (cl) vs p')
plt.xlabel('p')
plt.ylabel('cl')
plt.grid()

# Plot cl vs t
plt.subplot(2, 3, 3)
sns.scatterplot(data=df, x='t', y='cl', color='orange')
plt.title('Lift Coefficient (cl) vs t')
plt.xlabel('t')
plt.ylabel('cl')
plt.grid()

# Scatter plot for predicted vs actual cl values
plt.subplot(2, 3, 4)
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Diagonal line
plt.title('Predicted vs Actual cl Values')
plt.xlabel('Actual cl')
plt.ylabel('Predicted cl')
plt.xlim([y.min(), y.max()])
plt.ylim([y.min(), y.max()])
plt.grid()

# Residual plot
residuals = y_test - y_pred
plt.subplot(2, 3, 5)
sns.barplot(x=list(range(len(residuals))), y=residuals)
plt.title('Residuals (Actual - Predicted cl)')
plt.xlabel('Sample Index')
plt.ylabel('Residual')
plt.axhline(0, color='red', linestyle='--')
plt.grid()

# Histogram plot
plt.subplot(2, 3, 6)
sns.histplot(residuals, bins=30, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--')
plt.grid()

plt.tight_layout()
plt.show()

#%% MAKING PREDICTIONS
# Function to predict cl using user input for m, p, and t
def predict_cl(wu1, wu2, wu3, wl1, wl2, wl3):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[wu1, wu2, wu3, wl1, wl2, wl3]], columns=['wu1', 'wu2', 'wu3', 'wl1', 'wl2', 'wl3'])
    prediction = model.predict(input_data)
    return prediction[0]

# User input for predicting cl
print("\nEnter the parameters for the NACA airfoil to predict its cl value:")
try:
    wu1 = float(input("Enter WU1: "))
    wu2 = float(input("Enter WU2: "))
    wu3 = float(input("Enter WU3: "))
    wl1 = float(input("Enter WU1: "))
    wl2 = float(input("Enter WL2: "))
    wl3 = float(input("Enter WL3: "))

    
    predicted_cl = predict_cl(wu1, wu2, wu3, wl1, wl2, wl3)
    print(f"The predicted cl value for the given parameters (wu1={wu1}, wu2={wu2}, wu3={wu3}, wl1={wl1}, wl2={wl2}, wl3={wl3}) is: {predicted_cl:.4f}")
except ValueError:
    print("Invalid input! Please enter numeric values")

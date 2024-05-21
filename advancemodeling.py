import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import xgboost as xgb
import pandas as pd

# Read the CSV file
try:
    data = pd.read_csv("/Users/phonemyatmin/Downloads/Housing.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Assuming 'price' is the target variable based on your dataset's structure
X = data.drop('price', axis=1)
y = data['price']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boost Regressor': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'XGRF Regressor': xgb.XGBRFRegressor(),
    'Support Vector Regressor': SVR(),
    'Lasso Regressor': Lasso(),
    'Ridge Regressor': Ridge(),
    'Cat Boost Regressor': CatBoostRegressor(verbose=0)  # Disable CatBoost verbose output
}

predictions = {}
accuracy = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy[name] = {'MSE': mse, 'R2': r2}
    print(f"Results for {name}:")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    plt.figure(figsize=(15, 6))

    # Plot Actual vs. Predicted values
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(y_test)), y_test, label='Actual Trend')
    plt.plot(np.arange(len(y_test)), y_pred, label='Predicted Trend')
    plt.xlabel('Data')
    plt.ylabel('Trend')
    plt.legend()
    plt.title(f'Actual vs. Predicted for {name}')

    # Plot Residuals
    residuals = y_test - y_pred

    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {name}')

    plt.tight_layout()
    plt.savefig(f'output_comparison_{name.replace(" ", "_")}.png')
    plt.show()

# Create a DataFrame from the accuracy dictionary
accuracy_df = pd.DataFrame(accuracy).T
print(accuracy_df)

# Plot comparison of R2 scores
plt.figure(figsize=(10, 6))
accuracy_df['R2'].plot(kind='bar')
plt.ylabel('R2 Score')
plt.title('Comparison of R2 Scores for Different Models')
plt.savefig('r2_comparison.png')
plt.show()

# Plot comparison of MSE scores
plt.figure(figsize=(10, 6))
accuracy_df['MSE'].plot(kind='bar')
plt.ylabel('MSE Score')
plt.title('Comparison of MSE Scores for Different Models')
plt.savefig('mse_comparison.png')
plt.show()

# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Read the CSV file
try:
    data = pd.read_csv("/Users/phonemyatmin/Downloads/Housing.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Assuming 'price' is the target variable based on your dataset's structure
X = data.drop('price', axis=1)
y = data['price']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Define and fit the RFE model
rfe = RFE(estimator=LinearRegression(), n_features_to_select=10)
rfe.fit(X_train, y_train)

# Extract and print the names of the selected features
selected_columns = X_train.columns[rfe.support_]
X_train_rfe = X_train[selected_columns]
print("Selected Features:", selected_columns.tolist())

# Print data types (optional, for debugging)
print(X_train_rfe.dtypes)

# Function to build the regression model
def build_model(X, y):
    # Convert boolean data types to integers
    X = X.astype(int)
    
    # Add a constant to the model (for the intercept)
    X = sm.add_constant(X)
    
    # Fit the OLS model
    model = sm.OLS(y, X).fit()
    print(model.summary())  # Print the model summary
    return X

# Building the model using the RFE selected features
X_train_new = build_model(X_train_rfe, y_train)

# If 'bedrooms' is a feature in the new model, drop it
if "bedrooms" in X_train_new.columns:
    X_train_new = X_train_new.drop(["bedrooms"], axis=1)
    # Rebuild the model as Model 2 after dropping 'bedrooms'
    print("//Model 2")
    X_train_new = build_model(X_train_new, y_train)

# Function to check Variance Inflation Factor
def checkVIF(X):
    # Convert all columns to float to ensure VIF computation works
    X = X.astype(float)
    
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    return vif

# Checking VIF for the model
print("VIF Results:")
print(checkVIF(X_train_new))

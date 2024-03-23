import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl  # Import openpyxl for working with Excel files

import warnings
warnings.filterwarnings("ignore")

# Read the CSV file
try:
    df = pd.read_csv("/Users/phonemyatmin/Downloads/Housing.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("File not found. Please check the file path.")

# Display a sample of the DataFrame
print(df.sample(10))

# Data Stats
data_stats = df.describe()

# Encode categorical columns
df_encoded = pd.get_dummies(df)

# Calculate correlation
data_correlation = df_encoded.corr()

# Create a seaborn barplot
sns_plot = sns.barplot(x=df['airconditioning'], y=df['bedrooms'], hue=df["furnishingstatus"])

# Create a histogram of house prices
plt.figure(figsize=(10, 6))
plt.hist(df['price'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()

# Save the histogram as an image
plt.savefig("/Users/phonemyatmin/Downloads/price_distribution.png")

# Define the path to the output Excel file
output_excel_path = "/Users/phonemyatmin/Downloads/Housing_output.xlsx"

# Create an ExcelWriter object to write to the Excel file
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    # Write the DataFrame to a sheet named 'Data Information'
    df.to_excel(writer, sheet_name='Data Information', index=False)

    # Write the data stats to a new sheet named 'Data Stats'
    data_stats.to_excel(writer, sheet_name='Data Stats')

    # Write the correlation matrix to a new sheet named 'Data Correlation'
    data_correlation.to_excel(writer, sheet_name='Data Correlation')

    # Create a new sheet for seaborn plot and add the seaborn plot as an image
    sns_plot.figure.savefig("/Users/phonemyatmin/Downloads/seaborn_plot.png")
    worksheet = writer.book.create_sheet("Seaborn Plot")
    img = openpyxl.drawing.image.Image("/Users/phonemyatmin/Downloads/seaborn_plot.png")
    worksheet.add_image(img, 'B3')

    # Add the histogram of house prices as an image to a new sheet
    hist_worksheet = writer.book.create_sheet("Price Distribution")
    img = openpyxl.drawing.image.Image("/Users/phonemyatmin/Downloads/price_distribution.png")
    hist_worksheet.add_image(img, 'B3')

print(f"DataFrame, Data Stats, Data Correlation, and Histogram of House Prices saved to Excel file: {output_excel_path}")

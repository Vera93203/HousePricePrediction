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

"""
# Data Information
info_string = df.info()
print(info_string) 

"""

# Data Stats
data_stats = df.describe()
print(data_stats)

# Encode categorical columns
df_encoded = pd.get_dummies(df)

# Calculate correlation
data_correlation = df_encoded.corr()
print(data_correlation)

# Create a seaborn barplot
sns_plot = sns.barplot(x=df['airconditioning'], y=df['bedrooms'], hue=df["furnishingstatus"])

# Define the path to the output Excel file
output_excel_path = "/Users/phonemyatmin/Downloads/Housing_output.xlsx"

# Create an ExcelWriter object to write to the Excel file
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    # Write the DataFrame to the default sheet
    df.to_excel(writer, sheet_name='Data Information', index=False)


    # Write the data stats to a new sheet named 'Data Stats'
    data_stats.to_excel(writer, sheet_name='Data Stats')

    # Write the correlation matrix to a new sheet named 'Data Correlation'
    data_correlation.to_excel(writer, sheet_name='Data Correlation')

    # Create a new sheet for seaborn plot
    sns_plot.figure.savefig("/Users/phonemyatmin/Downloads/seaborn_plot.png")
    worksheet = writer.book.create_sheet("Seaborn Plot")
    img = openpyxl.drawing.image.Image("/Users/phonemyatmin/Downloads/seaborn_plot.png")
    worksheet.add_image(img, 'B3')

print(f"DataFrame and Data Stats saved to Excel file: {output_excel_path}")

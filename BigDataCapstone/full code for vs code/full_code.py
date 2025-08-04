# --------------------------------------------
# STEP 2: DATA CLEANING
# --------------------------------------------
# Author: Akize Israel
# Capstone Project - Big Data Analytics (Health Sector)
# Problem: Can we detect which hospitals have high patient wait times and why?
# Dataset: hospital_wait_times.csv
# --------------------------------------------

# --------------------------------------------
# Step 1: Import Libraries
# --------------------------------------------
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("Libraries Imported")

# --------------------------------------------
# Step 2: Load Dataset
# --------------------------------------------
df = pd.read_csv("data/hospital_wait_times.csv")

print("Dataset Shape (Rows x Columns):", df.shape)

print("\n----- First 5 Rows -----")
print(df.head())

# --------------------------------------------
# Step 3: Dataset Overview
# --------------------------------------------
print("\n----- Dataset Info -----")

print(df.info())

print("\n----- Summary Statistics -----")
print(df.describe(include='all'))

# --------------------------------------------
# Step 4: Check for Missing Values
# --------------------------------------------
print("\n----- Missing Values Per Column -----")
print(df.isnull().sum().sort_values(ascending=False))

# --------------------------------------------
# Step 5: Drop Irrelevant Columns (Optional)
# --------------------------------------------
columns_to_drop = ['Phone Number', 'ZIP Code', 'Address']
df.drop(columns=columns_to_drop, axis=1, inplace=True)
print(f"\n Dropped Irrelevant Columns: {columns_to_drop}")

# --------------------------------------------
# Step 6: Handle Missing Values
# --------------------------------------------
# Fill 'Emergency Services' missing with 'Unknown'
df['Emergency Services'].fillna('Unknown', inplace=True)

# Drop rows missing important identifiers
df.dropna(subset=['Location', 'Hospital Type'], inplace=True)
print("Missing Values Handled")

# --------------------------------------------
# Step 7: Standardize Column Names
# --------------------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("\n Standardized Column Names:")
print(df.columns.tolist())

# --------------------------------------------
# Step 8: Convert Ratings to Numeric
# --------------------------------------------
df.replace('Not Available', np.nan, inplace=True)
rating_cols = ['hospital_overall_rating']
for col in rating_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print("Converted rating columns to numeric")

# --------------------------------------------
# Step 9: Save Cleaned Dataset
# --------------------------------------------
df.to_csv("data/hospital_wait_times_cleaned.csv", index=False)
print("Cleaned Data Saved to: data/hospital_wait_times_cleaned.csv")

# --------------------------------------------
# Step 10: Final Dataset Status
# --------------------------------------------
print("\n----- Final Cleaned Dataset Info -----")
print(df.info())

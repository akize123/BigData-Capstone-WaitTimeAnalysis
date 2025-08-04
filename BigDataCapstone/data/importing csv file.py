import pandas as pd

# Load the dataset
df = pd.read_csv('data/importing csv file.csv')
print(df.describe())


# Preview first rows
print("----- HEAD -----")
print(df.head())

# Dataset summary
print("----- INFO -----")
print(df.info())

# Dataset shape
print("----- SHAPE -----")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# Dataset statistics
print("----- STATISTICS -----")

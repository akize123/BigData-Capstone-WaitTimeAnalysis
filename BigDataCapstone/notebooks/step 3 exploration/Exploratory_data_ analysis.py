

# --------------------------------------------
# Step 1: Import Libraries
# --------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# --------------------------------------------
# Step 2: Load Cleaned Dataset
# --------------------------------------------
df = pd.read_csv("data/hospital_wait_times_cleaned.csv")

print("Cleaned Dataset Loaded")
print("Dataset Shape:", df.shape)
print("\nColumn Types:")
print(df.dtypes)
print("\nSample Data:")
print(df.head())

# --------------------------------------------
# Step 3: Rating Distribution
# --------------------------------------------
plt.figure()
sns.histplot(df['hospital_overall_rating'].dropna(), bins=5, kde=True, color='teal')
plt.title("Distribution of Hospital Overall Ratings")
plt.xlabel("Rating (1 to 5)")
plt.ylabel("Number of Hospitals")
plt.tight_layout()
plt.show()

# --------------------------------------------
# Step 4: Emergency Services Count
# --------------------------------------------
plt.figure()
sns.countplot(data=df, x='emergency_services', palette='Set2')
plt.title("Availability of Emergency Services")
plt.xlabel("Emergency Services")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --------------------------------------------
# Step 5: Ratings by Ownership Type
# --------------------------------------------
plt.figure()
sns.boxplot(data=df, x='hospital_ownership', y='hospital_overall_rating', palette='Set3')
plt.title("Hospital Ratings by Ownership Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------
# Step 6: Ratings by State (Optional)
# --------------------------------------------
plt.figure()
sns.barplot(data=df, x='state', y='hospital_overall_rating', estimator='mean', ci=None, palette='viridis')
plt.title("Average Hospital Rating per State")
plt.xlabel("State")
plt.ylabel("Avg. Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------
# Summary
# --------------------------------------------
print("\n EDA Completed: Visuals Generated")

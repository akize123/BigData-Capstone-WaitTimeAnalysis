

# --------------------------------------------
# Step 1: Import Required Libraries
# --------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns





# --------------------------------------------
# Step 2: Load Cleaned Dataset
# --------------------------------------------
df = pd.read_csv("data/hospital_wait_times_cleaned.csv")

# --------------------------------------------
# Step 3: Select Features for Clustering
# --------------------------------------------
features = df[['hospital_overall_rating']].copy()
features['emergency_flag'] = df['emergency_services'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop rows with missing rating
features.dropna(inplace=True)

print("Features Selected for Clustering:")
print(features.head())

# --------------------------------------------
# Step 4: Normalize Features
# --------------------------------------------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --------------------------------------------
# Step 5: Determine Optimal Clusters (Elbow Method)
# --------------------------------------------
inertia = []
k_range = range(1, 6)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()

# --------------------------------------------
# Step 6: Apply KMeans with Optimal Clusters (e.g., K=3)
# --------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# --------------------------------------------
# Step 7: Visualize Clusters
# --------------------------------------------
plt.figure()
sns.scatterplot(data=df, x='hospital_overall_rating', y='emergency_flag', hue='cluster', palette='Set2', s=100)
plt.title("K-Means Clusters of Hospitals")
plt.xlabel("Hospital Rating")
plt.ylabel("Emergency Services (0 = No, 1 = Yes)")
plt.tight_layout()
plt.show()

# --------------------------------------------
# Step 8: Cluster Summary
# --------------------------------------------
print("\n Cluster Counts:")
print(df['cluster'].value_counts())

print("\n Clustering Complete")



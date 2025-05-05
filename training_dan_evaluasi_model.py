 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print("Data Loaded. Shape:", df.shape)
print(df.head())
# 2. Exploratory Data Analysis
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# 3. Preprocessing
df = df.drop('id', axis=1)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop target for unsupervised learning
df_no_target = df.drop('stroke', axis=1)

# Normalize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_no_target)

# === Training dan Evaluasi Model ===
# 1. Elbow Method (Evaluasi WCSS)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method - Menentukan jumlah klaster optimal')
plt.xlabel('Jumlah Klaster')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('elbow_method.png', dpi=300)
plt.close()
print("Elbow Method plot saved as 'elbow_method.png'.")

# 2. Silhouette Score (Evaluasi kualitas klaster)
print("\nSilhouette Scores:")
best_k = 2
best_score = -1
for n_clusters in range(2, 7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, cluster_labels)
    print(f"k = {n_clusters}, Silhouette Score = {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = n_clusters

print(f"\nBest k berdasarkan silhouette score: {best_k}")

# 3. Final Clustering dengan k optimal
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_data)

print("\nCluster counts:\n", df['cluster'].value_counts())

# 4. Simulasi Split Data (Training - Testing) untuk validasi internal (walau unsupervised)
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape}, Testing set: {X_test.shape}")

# Silhouette score pada training dan testing
train_labels = kmeans.fit_predict(X_train)
test_labels = kmeans.predict(X_test)
train_score = silhouette_score(X_train, train_labels)
test_score = silhouette_score(X_test, test_labels)
print(f"Train Silhouette Score: {train_score:.4f}")
print(f"Test Silhouette Score: {test_score:.4f}")

# 5. Ringkasan klaster
cluster_summary = df.groupby('cluster').mean()
print("\nCluster Summary (mean values per cluster):\n", cluster_summary)
cluster_summary.to_csv("cluster_summary.csv")
print("Cluster summary saved to 'cluster_summary.csv'.")

# 6. Visualisasi Clustering - PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='viridis', s=50)
plt.title('Visualisasi Klaster Pasien (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.savefig('cluster_pca_visualization.png', dpi=300)
plt.close()
print("Cluster visualization saved as 'cluster_pca_visualization.png'.")

 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load Data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print("Data Loaded. Shape:", df.shape)
print(df.head())

# 2. Exploratory Data Analysis
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# 3. Preprocessing

# Drop 'id' column (irrelevant for clustering)
df = df.drop('id', axis=1)

# Handle missing values (BMI: fill with median)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop target variable 'stroke' (since we use unsupervised learning)
df_no_target = df.drop('stroke', axis=1)

# Normalize numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_no_target)

print("\nData after preprocessing (first 5 rows):\n", pd.DataFrame(scaled_data, columns=df_no_target.columns).head())

# 4. Determine optimal number of clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_method.png', dpi=300)
plt.close()
print("Elbow Method plot saved as 'elbow_method.png'.")

# 5. Silhouette Score to evaluate cluster quality
for n_clusters in range(2, 7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is {silhouette_avg:.4f}")

# 6. Apply KMeans with optimal cluster count (assume k=3 based on analysis)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_data)

print("\nCluster counts:\n", df['cluster'].value_counts())

# 7. Cluster Analysis: Mean summary
cluster_summary = df.groupby('cluster').mean()
print("\nCluster Summary (means):\n", cluster_summary)

# 8. Visualizations

# 8.1 PCA for 2D Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='viridis', s=50)
plt.title('Cluster Visualization (PCA 2D)')
plt.savefig('cluster_pca_visualization.png', dpi=300)
plt.close()
print("Cluster visualization saved as 'cluster_pca_visualization.png'.")

# 8.2 Barplot: Jumlah anggota per klaster
plt.figure()
sns.countplot(x='cluster', data=df, palette='viridis')
plt.title('Jumlah Pasien per Klaster')
plt.xlabel('Klaster')
plt.ylabel('Jumlah Pasien')
plt.savefig('cluster_counts.png', dpi=300)
plt.close()
print("Barplot jumlah pasien per klaster disimpan sebagai 'cluster_counts.png'.")

# 8.3 Heatmap: Rata-rata fitur tiap klaster
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_summary, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Rata-rata Fitur per Klaster')
plt.savefig('cluster_feature_heatmap.png', dpi=300)
plt.close()
print("Heatmap cluster summary disimpan sebagai 'cluster_feature_heatmap.png'.")

# 9. Save Outputs
df.to_csv('clustered_patients.csv', index=False)
cluster_summary.to_csv('cluster_summary.csv')
print("Data dengan label klaster dan ringkasan klaster disimpan.")

# 10. Interpretation (Untuk Anda tulis di laporan, bukan bagian kode)
# - Klaster 0: Usia lebih tua, lebih banyak hipertensi dan penyakit jantung → potensi risiko tinggi
# - Klaster 1: Usia menengah, kondisi campuran
# - Klaster 2: Usia muda, sedikit komorbiditas → potensi risiko rendah

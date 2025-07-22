import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ===== STEP 1: Load & Merge Dataset =====
mat_df = pd.read_csv("student-mat.csv", sep=";")
por_df = pd.read_csv("student-por.csv", sep=";")

merge_columns = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]
merged_df = pd.merge(mat_df, por_df, on=merge_columns)

# Ambil kolom dari student-mat (_x)
cols_to_keep = merge_columns + [col for col in merged_df.columns if col.endswith('_x')]
df = merged_df[cols_to_keep]
df.columns = [col.replace('_x', '') for col in df.columns]

# ===== STEP 2: Siapkan Data Clustering =====
clustering_data = df[['absences', 'studytime']].copy()

# Standarisasi data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# ===== STEP 3: Clustering KMeans =====
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clustering_data['cluster'] = kmeans.fit_predict(scaled_data)

# Mapping cluster ke label deskriptif
cluster_labels = {
    0: "Siswa Disiplin & Tekun",
    1: "Siswa Berisiko",
    2: "Siswa Tengah / Sedang"
}

# Jika urutan label beda, kita hitung rerata untuk pastikan
cluster_summary = clustering_data.groupby('cluster')[['absences', 'studytime']].mean()
sorted_clusters = cluster_summary.sort_values(by=['absences', 'studytime'], ascending=[True, False]).index.tolist()
label_map = {
    sorted_clusters[0]: "Siswa Disiplin & Tekun",
    sorted_clusters[1]: "Siswa Tengah / Sedang",
    sorted_clusters[2]: "Siswa Berisiko"
}
clustering_data['segment'] = clustering_data['cluster'].map(label_map)

# ===== STEP 4: PCA untuk Visualisasi 2D =====
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
clustering_data['PC1'] = pca_components[:, 0]
clustering_data['PC2'] = pca_components[:, 1]

# ===== STEP 5: Visualisasi dengan Label Deskriptif =====
sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(10, 7))
palette = {'Siswa Disiplin & Tekun': 'green', 'Siswa Berisiko': 'red', 'Siswa Tengah / Sedang': 'orange'}
sns.scatterplot(
    data=clustering_data,
    x='PC1', y='PC2',
    hue='segment',
    palette=palette,
    s=80,
    edgecolor='black',
    alpha=0.8
)
plt.title("Segmentasi Siswa Berdasarkan Absensi dan Waktu Belajar", fontsize=16, fontweight='bold')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Segmentasi Siswa")
plt.tight_layout()
plt.show()

# ===== STEP 6: Ringkasan Tiap Segmen =====
print("\n📊 Rata-rata Absensi dan Waktu Belajar per Segmen:")
print(clustering_data.groupby('segment')[['absences', 'studytime']].mean().round(2))

# Simpan ke file jika diinginkan
clustering_data.to_csv("hasil_segmentasi_siswa.csv", index=False)
print("\n[✓] Data segmentasi disimpan sebagai 'hasil_segmentasi_siswa.csv'")

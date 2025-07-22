import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== STEP 1: Load dan Merge Data =====
mat_df = pd.read_csv("student-mat.csv", sep=";")
por_df = pd.read_csv("student-por.csv", sep=";")

print(f"📊 Ukuran Dataset:")
print(f"- Data Matematika: {mat_df.shape[0]} siswa")
print(f"- Data Portugis  : {por_df.shape[0]} siswa")

# Gabungkan data sesuai atribut umum
merge_columns = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]
merged_df = pd.merge(mat_df, por_df, on=merge_columns)
cols_to_keep = merge_columns + [col for col in merged_df.columns if col.endswith('_x')]
df = merged_df[cols_to_keep]
df.columns = [col.replace('_x', '') for col in df.columns]

print(f"- Data Gabungan  : {df.shape[0]} siswa")

# ===== STEP 2: Distribusi Jenis Kelamin =====
print("\n📌 Distribusi Jenis Kelamin:")
print(df['sex'].value_counts(normalize=True).mul(100).round(1))

plt.figure(figsize=(5,4))
sns.countplot(x='sex', data=df, palette='Set2')
plt.title("Distribusi Jenis Kelamin")
plt.xlabel("Jenis Kelamin")
plt.ylabel("Jumlah Siswa")
plt.tight_layout()
plt.show()

# ===== STEP 3: Distribusi Umur =====
plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=7, kde=True)
plt.title("Distribusi Umur Siswa")
plt.xlabel("Umur")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.show()

# ===== STEP 4: Status Sosial dan Pendidikan Orang Tua =====
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(x='Pstatus', data=df, ax=axes[0], palette='pastel')
axes[0].set_title("Status Pernikahan Orang Tua")
axes[0].set_xlabel("Pstatus (T=Together, A=Apart)")

sns.countplot(x='famsize', data=df, ax=axes[1], palette='muted')
axes[1].set_title("Ukuran Keluarga")
axes[1].set_xlabel("Ukuran (LE3=≤3, GT3=>3)")

plt.tight_layout()
plt.show()

# Pendidikan Orang Tua
plt.figure(figsize=(7,4))
sns.countplot(data=df, x='Medu', palette='Blues')
plt.title("Tingkat Pendidikan Ibu (Medu)")
plt.xlabel("0=Tidak sekolah, 4=Pendidikan Tinggi")
plt.tight_layout()
plt.show()

# ===== STEP 5: Distribusi Nilai G1, G2, G3 =====
nilai = df[['G1', 'G2', 'G3']]
nilai.plot(kind='box', figsize=(8,5), title="Distribusi Nilai G1, G2, G3")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ===== STEP 6: Absensi =====
plt.figure(figsize=(7,4))
sns.histplot(df['absences'], bins=30, kde=True, color='salmon')
plt.title("Distribusi Absensi")
plt.xlabel("Jumlah Absen")
plt.tight_layout()
plt.show()

# ===== STEP 7: Nilai vs Absensi =====
plt.figure(figsize=(6,5))
sns.scatterplot(x='absences', y='G3', data=df, alpha=0.6, edgecolor='black')
plt.title("Hubungan Absensi vs Nilai Akhir (G3)")
plt.xlabel("Absensi")
plt.ylabel("G3")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ===== STEP 8: Korelasi =====
print("\n🔍 Korelasi Nilai dan Variabel Penting:")
print(df[['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']].corr().round(2))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===== STEP 1: Load dan Gabungkan Dataset =====
mat_df = pd.read_csv("student-mat.csv", sep=";")
por_df = pd.read_csv("student-por.csv", sep=";")

merge_columns = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]
merged_df = pd.merge(mat_df, por_df, on=merge_columns)

# Ambil kolom _x dari student-mat
cols_to_keep = merge_columns + [col for col in merged_df.columns if col.endswith('_x')]
df = merged_df[cols_to_keep]
df.columns = [col.replace('_x', '') for col in df.columns]

# ===== STEP 2: Label Kelas Berdasarkan G3 =====
def label_grade(g3):
    if g3 >= 15:
        return "Tinggi"
    elif g3 >= 10:
        return "Sedang"
    else:
        return "Rendah"

df['grade_label'] = df['G3'].apply(label_grade)

# ===== STEP 3: Fitur & Target =====
features = df[['studytime', 'failures', 'absences']]
target = df['grade_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

# Standardisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== STEP 4: Model Klasifikasi (Random Forest) =====
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ===== STEP 5: Evaluasi Model =====
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n🧮 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ===== STEP 6: Visualisasi Hasil dengan PCA =====
# Gabungkan data untuk plot
X_all_scaled = scaler.transform(features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all_scaled)

# Buat DataFrame untuk visualisasi
vis_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
vis_df['Actual'] = target
vis_df['Predicted'] = model.predict(X_all_scaled)

# ===== Plot Visualisasi Klasifikasi (PCA) =====
sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(10, 7))
sns.scatterplot(data=vis_df, x="PC1", y="PC2", hue="Predicted", style="Actual", palette="Set1", s=80, edgecolor="black")
plt.title("Klasifikasi Siswa Berdasarkan Studytime, Failures, Absences", fontsize=15, fontweight="bold")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Prediksi / Realita", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

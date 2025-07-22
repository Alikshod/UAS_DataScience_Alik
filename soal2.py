import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ===== STEP 1: Load & Merge Dataset =====
mat_df = pd.read_csv("student-mat.csv", sep=";")
por_df = pd.read_csv("student-por.csv", sep=";")

merge_columns = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]
merged_df = pd.merge(mat_df, por_df, on=merge_columns)

# Ambil kolom _x saja (data dari student-mat)
cols_to_keep = merge_columns + [col for col in merged_df.columns if col.endswith('_x')]
df = merged_df[cols_to_keep]
df.columns = [col.replace('_x', '') for col in df.columns]

# ===== STEP 2: Fungsi Visualisasi Regresi =====
sns.set(style="whitegrid", palette="pastel")

def plot_regression(df, x_col, y_col='G3', color='blue', file_name=None, title_suffix='', show=True, save=True):
    model = LinearRegression()
    X = df[[x_col]]
    y = df[y_col]
    model.fit(X, y)

    coef = model.coef_[0]
    intercept = model.intercept_

    plt.figure(figsize=(7, 5))
    sns.regplot(
        x=x_col,
        y=y_col,
        data=df,
        ci=None,
        scatter_kws={'alpha': 0.6, 'color': color},
        line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 2}
    )

    plt.title(f"Regresi Linear: {x_col} vs {y_col} {title_suffix}", fontsize=14, fontweight='bold')
    plt.xlabel(x_col.capitalize(), fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)

    # Tampilkan persamaan regresi di pojok kiri atas
    plt.text(
        0.05, 0.95,
        f'y = {coef:.2f}x + {intercept:.2f}',
        ha='left', va='top',
        transform=plt.gca().transAxes,
        fontsize=11,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )

    plt.tight_layout()

    # Tampilkan grafik
    if show:
        plt.show()

    # Simpan grafik ke file
    if save and file_name:
        plt.savefig(file_name, dpi=300)
        print(f"[✓] Grafik '{file_name}' berhasil disimpan.")

    plt.close()

# ===== STEP 3: Buat dan Tampilkan Kedua Grafik =====
plot_regression(df, 'studytime', color='skyblue', file_name='reg_studytime_g3.png', title_suffix='(Waktu Belajar)')
plot_regression(df, 'absences', color='salmon', file_name='reg_absences_g3.png', title_suffix='(Absensi)')

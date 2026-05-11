#%%
from loadingData import df
from config import sns,plot,stats,PCA,pd,KNNImputer
#DATA CLEANSING

# %%
# Menghitung jumlah duplikat
duplicates = df.duplicated().sum()
print(duplicates)

# %%
# Mengidentifikasi baris duplikat
print(df[df.duplicated()])

# %%
# Menghapus duplikat
df_cleaned = df.drop_duplicates()

# %%
# Cek nilai unik kolom kategorikal
df['branch'].unique()
df['company_type'].unique()
df['job_role'].unique()

# %%
# Standarisasi format string (huruf kecil semua)
df['branch'] = df['branch'].str.lower()
df['company_type'] = df['company_type'].str.lower()

# %%
# Ganti nilai yang tidak konsisten
df['branch'] = df['branch'].str.strip()  # hapus spasi di awal/akhir

# %%
# Cek nilai tidak valid, misal cgpa tidak mungkin > 10 atau < 0
df[df['cgpa'] > 10]
df[df['cgpa'] < 0]

# %%
# Ganti nilai tidak valid dengan NaN
df['cgpa'] = df['cgpa'].apply(lambda x: x if 0 <= x <= 10 else None)

# %%
# BOXPLOT - deteksi outlier secara visual
sns.boxplot(x=df['salary_lpa'])
plot.title('Boxplot Salary LPA')
plot.show()

# %%
# Z-SCORE - outlier jika z-score > 3 atau < -3
df['z_score'] = stats.zscore(df['salary_lpa'])
outliers_zscore = df[(df['z_score'] > 3) | (df['z_score'] < -3)]
print("Outliers Z-Score:\n", outliers_zscore)

# %%
# IQR - outlier jika di luar Q1-1.5*IQR atau Q3+1.5*IQR
Q1 = df['salary_lpa'].quantile(0.25)
Q3 = df['salary_lpa'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['salary_lpa'] < (Q1 - 1.5 * IQR)) | (df['salary_lpa'] > (Q3 + 1.5 * IQR))]
print("Outliers IQR:\n", outliers_iqr)

# %%
# Menghapus outlier berdasarkan Z-Score
df_cleaned = df[df['z_score'] <= 3]

#DATA REDUCTION
# %%

# Pilih kolom numerik untuk PCA
X = df[['cgpa', 'coding_score', 'communication_score', 
        'aptitude_score', 'resume_score', 'skill_score']]

# %%
# Terapkan PCA, kurangi dimensi jadi 2 komponen
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Data Original Shape:", X.shape)
print("Data Setelah PCA Shape:", X_pca.shape)
print("PCA Transformed Data:\n", X_pca)


#%%
#Handling Missing Value
# %%
# Cek missing value per kolom
print(df.isnull().sum())

# %%
# Cek missing value dalam bentuk boolean
print(df.isnull())

# %%
# Cek non-missing value
print(df.notnull())

# %%
# Info jumlah non-null tiap kolom
df.info()

# %%
# Visualisasi missing value dengan heatmap
plot.figure(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plot.title('Visualisasi Missing Data')
plot.xlabel('Kolom')
plot.ylabel('Baris')
plot.show()

# %%
# Menghapus baris dengan missing value
df_dropped_rows = df.dropna()
print(df_dropped_rows)

# %%
# Menghapus kolom dengan missing value
df_dropped_cols = df.dropna(axis=1)
print(df_dropped_cols)

# %%
# Mengisi missing value dengan mean
df_filled_mean = df.fillna(df.select_dtypes(include='number').mean())

# %%
# Mengisi missing value dengan median
df_filled_median = df.fillna(df.select_dtypes(include='number').median())

# %%
# Mengisi missing value dengan modus
df_filled_mode = df.fillna(df.mode().iloc[0])

# %%

# Inisialisasi KNN Imputer
imputer = KNNImputer(n_neighbors=2)

# Mengisi missing value dengan KNN
df_knn_imputed = pd.DataFrame(imputer.fit_transform(df[['cgpa', 'coding_score', 
                'communication_score', 'aptitude_score', 
                'resume_score', 'skill_score', 'salary_lpa']]), 
                columns=['cgpa', 'coding_score', 'communication_score', 
                'aptitude_score', 'resume_score', 'skill_score', 'salary_lpa'])
print(df_knn_imputed)

#%%
#DATA TRANSFORMATION
# %%
# One-Hot Encoding pada kolom kategorikal
df_encoded = pd.get_dummies(df, columns=['branch', 'company_type', 'job_role'])
print("Data Sebelum Encoding:")
print(df.head())
print("\nData Setelah One-Hot Encoding:")
print(df_encoded.head())

# %%
# Menghitung total dan rata-rata salary per branch
agregasi = df.groupby('branch')['salary_lpa'].sum().reset_index()
print("Total Salary per Branch:")
print(agregasi)

# %%
# Menghitung rata-rata CGPA per company type
agregasi_mean = df.groupby('company_type')['cgpa'].mean().reset_index()
print("Rata-rata CGPA per Company Type:")
print(agregasi_mean)

# %%
# Normalisasi Min-Max pada kolom numerik
df['cgpa_normalized'] = (df['cgpa'] - df['cgpa'].min()) / (df['cgpa'].max() - df['cgpa'].min())
df['salary_normalized'] = (df['salary_lpa'] - df['salary_lpa'].min()) / (df['salary_lpa'].max() - df['salary_lpa'].min())

print("Data Setelah Normalisasi Min-Max:")
print(df[['cgpa', 'cgpa_normalized', 'salary_lpa', 'salary_normalized']])

# %%
# Standardisasi Z-Score pada kolom numerik
mean = df['cgpa'].mean()
std_dev = df['cgpa'].std()
df['cgpa_zscore'] = (df['cgpa'] - mean) / std_dev

mean_sal = df['salary_lpa'].mean()
std_sal = df['salary_lpa'].std()
df['salary_zscore'] = (df['salary_lpa'] - mean_sal) / std_sal

print("Data Setelah Standardisasi Z-Score:")
print(df[['cgpa', 'cgpa_zscore', 'salary_lpa', 'salary_zscore']])

# %%
# Discretization pada kolom salary_lpa
bins = [0, 10, 30, float('inf')]
labels = ['Rendah', 'Sedang', 'Tinggi']

df['salary_kategori'] = pd.cut(df['salary_lpa'], bins=bins, labels=labels, right=False)

print("Data Sebelum Discretization:")
print(df['salary_lpa'])
print("\nData Setelah Discretization:")
print(df[['salary_lpa', 'salary_kategori']])

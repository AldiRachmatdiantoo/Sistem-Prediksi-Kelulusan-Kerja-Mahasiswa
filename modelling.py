#%%
# # MODELLING DATA - Student Placement & Salary


#%% 1. IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.impute import KNNImputer

# Regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix,
                             classification_report)
import statsmodels.api as sm

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

# Konfigurasi
warnings.filterwarnings('ignore')

print("✅ Semua library berhasil diimport.")

#%% 2. LOAD & PREPROCESSING DATA
df = pd.read_csv('student_placement_salary_elite_v2.csv')
print("Shape data awal:", df.shape)

# Clean Duplicates & Standarisasi String
df = df.drop_duplicates()
for col in ['branch', 'company_type', 'job_role']:
    if col in df.columns:
        df[col] = df[col].str.lower().str.strip()

# Validasi CGPA & Imputasi
if 'cgpa' in df.columns:
    df['cgpa'] = df['cgpa'].apply(lambda x: x if 0 <= x <= 10 else np.nan)

num_cols = ['cgpa', 'coding_score', 'communication_score', 'aptitude_score', 'salary_lpa']
num_cols = [c for c in num_cols if c in df.columns]

imputer = KNNImputer(n_neighbors=2)
df[num_cols] = imputer.fit_transform(df[num_cols])

# Hapus Outlier
df = df[np.abs(stats.zscore(df['salary_lpa'])) <= 3]

print(f"✅ Preprocessing Selesai. Shape: {df.shape}")
df.head()

#%% 3. FEATURE ENGINEERING (ENCODING)
le = LabelEncoder()
df_model = df.copy()

categorical_cols = ['branch', 'company_type', 'job_role']
for col in categorical_cols:
    if col in df_model.columns:
        df_model[col + '_enc'] = le.fit_transform(df_model[col].astype(str))

# Definisi Fitur
feature_cols = [c for c in ['cgpa', 'coding_score', 'communication_score', 
                             'aptitude_score', 'branch_enc', 'job_role_enc'] 
                if c in df_model.columns]

X = df_model[feature_cols]
y = df_model['salary_lpa']

print("Fitur yang digunakan:", feature_cols)

#%% 4. REGRESSION MODELLING (LINEAR & LASSO)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Lasso Regression
lasso_model = Lasso(alpha=0.1).fit(X_train, y_train)
y_pred_ls = lasso_model.predict(X_test)

print(f"Linear R² : {r2_score(y_test, y_pred_lr):.4f}")
print(f"Lasso R²  : {r2_score(y_test, y_pred_ls):.4f}")

#%% 5. LOGISTIC REGRESSION (CLASSIFICATION)
# Target: Gaji di atas median (High Salary)
median_val = df_model['salary_lpa'].median()
y_bin = (df_model['salary_lpa'] > median_val).astype(int)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_bin, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000).fit(X_train_b, y_train_b)
y_pred_log = log_reg.predict(X_test_b)

print("--- Classification Report Logistic Regression ---")
print(classification_report(y_test_b, y_pred_log))

#%% 6. CLUSTERING (K-MEANS)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[['cgpa', 'coding_score', 'salary_lpa']])

# Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#%% 7. AGNES & DBSCAN
# AGNES
agnes_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

print(f"Silhouette Score AGNES: {silhouette_score(X_scaled, agnes_labels):.4f}")
if len(set(db_labels)) > 1:
    print(f"DBSCAN Clusters: {len(set(db_labels)) - (1 if -1 in db_labels else 0)}")

#%% 8. VISUALIZATION (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agnes_labels, cmap='viridis', alpha=0.7)
plt.title('Clustering Results (PCA Projection)')
plt.colorbar(label='Cluster ID')
plt.show()
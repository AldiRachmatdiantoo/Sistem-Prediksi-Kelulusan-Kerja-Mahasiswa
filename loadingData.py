#%%
# Import config.py
from config import pd

#%%
# DATA COLLECTION
df = pd.read_csv('student_placement_salary_elite_v2.csv')

#%%
# LOADING DATA

#%%
df.info()
#%%
df.head()
#%%
df.describe()
#%%
df.describe(include='object')
#%%
df.columns
# %%
df.shape
# %%
df.dtypes
# %%
df['branch'].unique()
# %%
df['branch'].nunique()
# %%
df['cgpa'].value_counts()
#%%
df['cgpa'].mean()
#%%
df['cgpa'].median()
#%%
df['cgpa'].mode()


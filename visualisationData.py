#%%
#Histogram
from config import plot,sns
from loadingData import df

sns.histplot(df['cgpa'], bins=20, kde=True)
plot.title('Distribusi CGPA')
plot.xlabel('CGPA')
plot.ylabel('Count')

#%%
#Box Plot
sns.boxplot(x=df['salary_lpa'])
plot.title('Boxplot Salary LPA')
plot.show()

# %%
# LINE CHART - tren data
sns.lineplot(x=df['cgpa'], y=df['salary_lpa'])
plot.title('Tren CGPA vs Salary')
plot.xlabel('CGPA')
plot.ylabel('Salary LPA')
plot.show()

# %%
# SCATTER PLOT - hubungan dua variabel numerik
sns.scatterplot(x=df['cgpa'], y=df['salary_lpa'])
plot.title('Scatter Plot CGPA vs Salary')
plot.xlabel('CGPA')
plot.ylabel('Salary LPA')
plot.show()


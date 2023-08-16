# libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#!pip install hvplot
#import hvplot.pandas

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

%matplotlib inline
df = pd.read_csv('/kaggle/input/tesla-2015-2022/TSLA.csv')
df.head()
df.info()
df.describe()
df.value_counts('Open')
df.value_counts('High')
df.value_counts('Low')
sns.histplot(df, x ='High')
plt.show()
sns.histplot(df, x ='High', binwidth = 10)
plt.show()
sns.histplot(df, x ='Low')
plt.show()
sns.histplot(df, x='Low', binwidth = 10)
plt.show()
sns.histplot(df, x ='Open')
plt.show()
sns.histplot(df, x='Open', binwidth = 10)
plt.show()
sns.histplot(df, x ='Close')
plt.show()
sns.histplot(df, x='Close', binwidth = 10)
plt.show()
df.dtypes
df['Open'] = df['Open'].astype(int)
df['Low'] = df['Low'].astype(int)
df['Close'] = df['Close'].astype(int)
df['Adj Close'] = df['Adj Close'].astype(int)
df['Volume'] = df['Volume'].astype(int)
df.dtypes
df['Open'].isin(['High', 'Low'])
df['Close'].isin(['High', 'Low'])
df['Open'].max()
df['Open'].min()
df['Close'].max()
df['Close'].min()
sns.boxplot(df, x='Open')
plt.show()
sns.boxplot(df, x='Close')
plt.show()

df.groupby('Volume').mean()
df.groupby('Volume').sum()
df.groupby('Volume').count()
df.groupby('Volume').min()
df.groupby('Volume').max()
df.groupby('Volume').var()
df.groupby('Volume').std()
df.agg(['mean','count', 'min', 'max', 'var', 'std', 'median'])
sns.barplot(df, x='Open', y='Close')
plt.show()
sns.barplot(df, x='High', y='Low')
plt.show()
sns.heatmap(df.corr(), annot=True,cmap='Reds')
sns.pairplot(df)

















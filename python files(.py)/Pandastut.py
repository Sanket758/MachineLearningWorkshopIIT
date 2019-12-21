#!/usr/bin/env python
# coding: utf-8
import pandas as pd
data=pd.read_csv("C:/Users/DELL/Downloads/Titanic_Data.csv")
type(data)
data.tail()
data.head(10)
data.isna().sum()
data.columns
data.describe()
data.isnull().sum()
data.isna().sum()
data.info()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
sns.heatmap(data.isnull())

data.dropna()

data.drop(['PassengerId'],axis=1)

data.std()

data.sum()

data.mean()

data['Age'].median()

data['Age'].mean()

data['Sex']





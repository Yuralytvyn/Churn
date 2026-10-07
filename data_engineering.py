#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# In[2]:


df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df


# In[3]:


df.dtypes


# ## Converting Columns to Numeric Format

# In[4]:


for col in df:
    if df[col].dtype == "str":
        df[col] = df[col].astype("category").cat.codes


# ## Data analysis

# In[5]:


df.isna().sum()


# In[6]:


df.corr()["Churn"]


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")

plt.show()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

correlation = df.corr(numeric_only=True)["Churn"].sort_values(ascending=False)

sns.heatmap(correlation.to_frame(), annot=True, cmap="coolwarm")

plt.show()


# ## Data engineering

# In[9]:


df = df.drop(columns=["customerID"])


# In[9]:

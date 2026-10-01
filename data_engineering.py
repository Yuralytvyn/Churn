#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df1 = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[3]:


df.dtypes


# In[4]:


df["PaymentMethod"].unique()


# ## Converting Columns to Numeric Format

# In[5]:


df["id"] = range(len(df))


# In[6]:


df["gender"] = df["gender"].map({
    "Female": 0,
    "Male": 1
})


# In[7]:


df["Partner"] = df["Partner"].map({
    "Yes": 1,
    "No": 0
})


# In[8]:


df["Dependents"] = df["Dependents"].map({
    "Yes": 1,
    "No": 0
})


# In[9]:


df["PhoneService"] = df["PhoneService"].map({
    "Yes": 1,
    "No": 0
})


# In[10]:


df["MultipleLines"] = df["MultipleLines"].map({
    "No phone service":2,
    "Yes": 1,
    "No": 0
})


# In[11]:


df["InternetService"] = df["InternetService"].map({
    "DSL":2,
    "Fiber optic": 1,
    "No": 0
})


# In[12]:


df["OnlineSecurity"] = df["OnlineSecurity"].map({
    "No internet service":2,
    "Yes": 1,
    "No": 0
})


# In[13]:


df["OnlineBackup"] = df["OnlineBackup"].map({
    "No internet service":2,
    "Yes": 1,
    "No": 0
})


# In[14]:


df["DeviceProtection"] = df["DeviceProtection"].map({
    "No internet service":2,
    "Yes": 1,
    "No": 0
})


# In[15]:


df["TechSupport"] = df["TechSupport"].map({
    "No internet service":2,
    "Yes": 1,
    "No": 0
})


# In[16]:


df["StreamingTV"] = df["StreamingTV"].map({
    "No internet service":2,
    "Yes": 1,
    "No": 0
})


# In[17]:


df["StreamingMovies"] = df["StreamingMovies"].map({
    "No internet service":2,
    "Yes": 1,
    "No": 0
})


# In[18]:


df["Contract"] = df["Contract"].map({
    "Month-to-month":2,
    "Two year": 1,
    "One year": 0
})


# In[19]:


df["PaperlessBilling"] = df["PaperlessBilling"].map({
    "Yes": 1,
    "No": 0
})


# In[19]:





# In[3]:


import pandas as pd
df1 = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
for col in df1:
    if df1[col].dtype == "str":
        df1[col] = df1[col].astype("category").cat.codes
df1


# In[8]:


df1.corr()["Churn"]


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))

sns.heatmap(
    df1.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.show()


# In[21]:





# In[21]:





# In[21]:





# In[21]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns 



# # Data import 

# In[4]:


data = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"


# In[5]:


df = pd.read_csv(data)
df.head(10)


# In[9]:


display(df.shape)


# In[10]:


df.columns


# In[11]:


df.dtypes


# In[12]:


sns.histplot(df.median_house_value)


# In[14]:


sns.histplot(np.log1p(df.median_house_value.values))


# In[15]:


df = df[["longitude", "latitude", "housing_median_age", "total_rooms","total_bedrooms", "population", "households", 
         "median_income", "median_house_value"]]
df


# # Question 1

# In[17]:


df.isnull().sum()


# # Question 2

# In[18]:


df['population'].describe()


# In[19]:


df['population'].median()


# In[21]:


n = len(df)
n_val = int(n*0.2)
n_test = int(n*0.2)
n_train = n-n_val-n_test

n_val, n_test, n_train


# In[23]:


idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
idx


# In[24]:


df.iloc[idx[:10]]


# In[25]:


df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]


# In[26]:


df_train


# In[28]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[31]:


y_train = np.log1p(df_train.median_house_value.values)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)


# In[32]:


del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']


# In[33]:


len(y_train)


# # Question 3

# In[34]:


def prepare_X_fill_with_zeros(df):
    df = df.copy()
    df = df.fillna(0)
    X = df.values
    return X


# In[ ]:





# In[35]:


def prepare_X_fill_with_mean(df):
    df = df.copy()
    df = df.fillna(df['total_bedrooms'].mean())
    X = df.values
    return X


# In[36]:


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


# In[37]:


def rmse(y_pred, y):
    se = (y_pred - y)**2
    mse = se.mean()
    return np.sqrt(mse)


# In[39]:


X_train = prepare_X_fill_with_zeros(df_train)
w0, w = train_linear_regression(X_train, y_train)


X_val = prepare_X_fill_with_zeros(df_val)
y_pred = w0 + X_val.dot(w)

round(rmse(y_pred, y_val), 2)


# In[40]:


X_train = prepare_X_fill_with_mean(df_train)
w0, w = train_linear_regression(X_train, y_train)

#validation part
X_val = prepare_X_fill_with_mean(df_val)
y_pred = w0 + X_val.dot(w)

round(rmse(y_pred, y_val), 2)


# # Question 4

# In[41]:


def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


# In[47]:


for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    
    X_train = prepare_X_fill_with_zeros(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    
    X_val = prepare_X_fill_with_zeros(df_val)
    y_pred = w0 + X_val.dot(w)

    score = rmse(y_pred, y_val)

    print(r, w0, round(score, 2))
    


# In[53]:


RMSE_set = []

for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:

    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test
    idx = np.arange(n)
    
    np.random.seed(s)
    np.random.shuffle(idx)
    
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]
    
     
    df_train = df_train.reset_index(drop=True)
   
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    
    y_train = np.log1p(df_train.median_house_value.values)
   
    y_val = np.log1p(df_val.median_house_value.values)
    y_test = np.log1p(df_test.median_house_value.values)
    
    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']
    X_train = prepare_X_fill_with_zeros(df_train)
    w0, w = train_linear_regression(X_train, y_train)
    # validation part
    X_val = prepare_X_fill_with_zeros(df_val)
    y_pred = w0 + X_val.dot(w)

    rmse_result = rmse(y_pred, y_val)
    print('RMSE for seed = %s:' % s, rmse_result)
    RMSE_set.append(rmse_result)


# In[54]:


print(RMSE_set)
print(len(RMSE_set))

std = np.std(RMSE_set)
print('standard deviation of all the scores', std)


# In[ ]:





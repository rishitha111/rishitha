
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataframe = pd.read_csv('winequality-red.csv')


# In[3]:


dataframe.info()


# In[5]:


features = dataframe[['citric acid','fixed acidity','density','pH','alcohol']]


# In[6]:


target = dataframe['quality']


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.hist(target,bins=10)


# In[9]:


import numpy as np


# In[10]:


dataframe_normalized = np.log(target)


# In[11]:


plt.hist(dataframe_normalized)


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train_n,X_test_n,y_train_n,y_test_n = train_test_split(features,dataframe_normalized,test_size=0.2,train_size=0.8)


# In[14]:


from sklearn.metrics import r2_score


# In[15]:


from sklearn.linear_model import LinearRegression


# In[17]:


regressor = LinearRegression()
reg_fit = regressor.fit(X_train_n,y_train_n)
reg_pred= reg_fit.predict(X_test_n)


# In[19]:


score_norm = r2_score(y_test_n,reg_pred)


# In[20]:


print(score_norm)


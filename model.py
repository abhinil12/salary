#!/usr/bin/env python
# coding: utf-8

# In[3]:


## importing libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


dataset = pd.read_csv('hiring.csv')


# In[5]:


dataset['experience'].fillna(0, inplace=True)


# In[6]:


dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)


# In[7]:


X = dataset.iloc[:, :3]


# In[8]:


#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]


# In[9]:


X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))


# In[10]:


y = dataset.iloc[:, -1]


# In[ ]:


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.


# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[12]:


#Fitting model with trainig data
regressor.fit(X, y)


# In[13]:


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


# In[14]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # EDA performed on a dataset related to RED WINE

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Data preprocessing part
# 

# In[2]:


# Loading the dataset
df = pd.read_csv('winequality-red.csv')


# In[3]:


# Seeing the dataset
df


# In[4]:


df.info()


# See the 'Non-Null Count' column - No null values in any columns  
# Total rows or entries we have = 1599  
# Also the dtpyes or data-types for all the columns is float, only the quality is int

# In[5]:


df_desc = df.describe()
df_desc


# In[6]:


df_description = df.describe().transpose()
df_description


# ### Pulling out some statistics from the dataset

# Found out from the given dataset from kaggle that...  
# Quality > 6.5 should be considered as good

# In[7]:


# Finding out the minimum
min_quality = np.min(df['quality'])
max_quality = np.max(df['quality'])


mean_quality = df_desc['quality']['mean']
mean_quality = round(mean_quality, 2)

print(f"The min quality of wine present in the dataset is {min_quality}")
print(f"The max quality of wine present in the dataset is {max_quality}")
print(f"Mean value of the quality of wine is {mean_quality}")


# In[8]:


std_df = df_description['std']
std_df


# In[9]:


max_std_feature = std_df.idxmax()
min_std_feature = std_df.idxmin()

print(f"Feature with the maximum std: {max_std_feature}")
print(f"Feature with the minimum std: {min_std_feature}")


# In[15]:


corr_df = df.corr()
print("Here is the correlation:")
corr_df


# In[18]:


alc_corr_df = corr_df['quality']
alc_corr_df


# Features that are having a positive correlation with the quality level - fixed acidity, citric acid, residual sugar, sulphates, alcohol   
# There are so many features having negative correlation with the alchol as well. But there a few features which are have a very higher negative correlation with the quality comparitively - total sulfur dioxide, chorides, density, volatile acidity
#  
# 
# But it doesnt matter whether positive or negative - a high correlation means a high correlation - it will  provide valuable impact on model.

# ### Plotting some plots and graphs to visualize the relationship between various features
# 

# In[10]:


df


# In[11]:


feature_set1 = ['pH', 'sulphates', 'alcohol']
df_pairplot_set1 = df[feature_set1]
df_pairplot_set1


# In[12]:


# Plotting pairplot on feature set - 1
pplot_fs1 = sns.pairplot(df_pairplot_set1)


# Linear relation between alcohol and pH level can be seen.  
# Linear relation can be seen in the scatterplot of alcohol vs sulphates - although there are remarkable amount of outliers...

# In[13]:


# Since pH has a very good linear relation with the alchol
# For the second set of feature let's puick up pH, citric acid, and density
feature_set2 = ['pH', 'density', 'citric acid']

df_pairplot_set2 = df[feature_set2]
df_pairplot_set2


# In[22]:


# Plotting pairplot on feature set - 2
pplot_fs2 = sns.pairplot(df_pairplot_set2)


# A positive slope relation can be seen between density vs citric acid scatterplot.  
# Linear relation can be seen in the plot of citric acid and density.  
# 

# In[23]:


df


# In[24]:


# Let's see what is the relation between the quality, alcohol and volatile acidity
feature_set3 = ['quality', 'alcohol', 'volatile acidity']
df_pairplot_set3 = df[feature_set3]
df_pairplot_set3


# In[ ]:


pplot_fs3 = sns.pairplot()


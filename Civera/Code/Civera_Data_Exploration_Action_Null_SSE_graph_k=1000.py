#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_desc_null_actions = pd.read_csv("df_null_action.csv", nrows=10000)
df_desc_null_actions['2']


# In[3]:


# Vectorize the descriptions.
tfidf = TfidfVectorizer(min_df=5, ngram_range=(1,3))

# features = tfidf.fit_transform(dataset['0'].astype('U'))
description_mia_vectorized = tfidf.fit_transform(df_desc_null_actions['2'].astype('U'))


# In[4]:


description_mia_vectorized


# In[5]:


# Create a kmeans model on our data, using k clusters. Compile a list of SSEs for each k.

k_list = []
SSE_list = []

number_clusters = range(800,1000)

for i in number_clusters:
    kmeans_model = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                         n_clusters=i, n_init=10, random_state=None, tol=0.0001, verbose=0).fit(description_mia_vectorized)
    SSE = kmeans_model.inertia_
    SSE_list.append(SSE)
    k_list.append(i)


# In[6]:


# Plot the results
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(k_list, SSE_list)
ax1.set_xlabel('$No. of Clusters$')
ax1.set_ylabel('$SSE$')


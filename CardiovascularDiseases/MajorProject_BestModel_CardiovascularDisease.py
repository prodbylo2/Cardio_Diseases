#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv("cardio_train.csv", sep=";")
df = df.drop(['id'], axis = 1)
df


# In[6]:


plt.figure(figsize=(20,18), dpi= 80)
sns.pairplot(df, kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()


# In[7]:


df.corr()


# In[8]:


x = df.iloc[:, 0:11]
x


# In[9]:


y = df.iloc[:, 11:]
y


# In[25]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.4)


# **DECISION TREE**

# In[26]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# fitting the model
dtc = DecisionTreeClassifier()
dtc.fit(xtrain, ytrain)

# prediction
ans = dtc.predict(xtest)
ans

# plotting the tree
tree.plot_tree(dtc)


# In[46]:


# score array
score_arr = []


# In[47]:


# score of decision tree model
score_arr.append(dtc.score(xtest, ytest))


# **RANDOM FOREST CLASSIFIER**

# In[48]:


from sklearn.ensemble import RandomForestClassifier

rm = RandomForestClassifier()
rm.fit(xtrain, ytrain)

# prediciton
rm.predict(xtest)


# In[49]:


# score of random forest classifier model
score_arr.append(rm.score(xtest, ytest))


# **LOGISTIC REGRESION**

# In[50]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()

# fitting the model
log.fit(xtrain, ytrain)

# prediction
log.predict(xtest)


# In[51]:


# score of logistic regression model
score_arr.append(log.score(xtest, ytest))


# **SUPPORT VECTOR MACHINE**

# In[71]:


from sklearn.svm import SVC
import numpy as np

sv = SVC() 

# because ytrain is a column-vector or 2D array
ytrain_1d = np.ravel(ytrain)

sv.fit(xtrain, ytrain_1d)

sv.predict(xtest)


# In[72]:


# score of support vector machine
score_arr.append(sv.score(xtest, ytest))


# **K-NEAREST NEIGHBOUR**

# In[76]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(xtrain, ytrain)

knn.predict(xtest)


# In[77]:


# score of knn classification algorithm
score_arr.append(knn.score(xtest, ytest))


# In[78]:


score_arr


# **As we can see the model that produces the most accurate result is the random forest classifier**

# -------------------------------------------------------------------------------------------------------------------------------

# In[ ]:





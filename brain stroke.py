#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv('brain_stroke.csv')
data.head
data.dtypes


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.svm import SVC

df = pd.DataFrame(data)


# In[4]:


data.dtypes


# In[5]:


data = pd.get_dummies(data,columns=['gender', 'ever_married', 'work_type' , 'Residence_type' , 'smoking_status'],drop_first=True)


# In[6]:


data.dtypes


# In[7]:


X = data.drop(columns=['stroke'])
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# In[8]:


from sklearn.svm import SVC


# In[14]:


svclassifier = SVC(kernel= 'sigmoid')
svclassifier.fit(X_train, y_train)


# In[15]:


from sklearn.metrics import accuracy_score
y_pred = svclassifier.predict(X_test)

score = accuracy_score(y_test, y_pred)
score


# In[12]:


X = data.drop(columns=['stroke'])
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion="entropy", random_state= 1000, max_depth= 4, min_samples_leaf = 1)
model.fit(X_train, y_train)
decTreeRegPredict = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
score #91.85


# In[ ]:


X = data.drop(columns=['stroke'])
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn250 = KNeighborsClassifier (n_neighbors=2500)
knn250.fit (X_train , y_train)
knn250classification = knn250.predict (X_test)
knn250classification
knn250accuraancy = accuracy_score (knn250classification , y_test)
knn250accuraancy


# In[ ]:


X = data.drop(columns=['stroke'])
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn2500 = KNeighborsClassifier (n_neighbors=3600)
knn2500.fit (X_train , y_train)
knn2500classification = knn2500.predict (X_test)
knn2500classification
knn2500accuraancy = accuracy_score (knn2500classification , y_test)
knn2500accuraancy


# In[ ]:





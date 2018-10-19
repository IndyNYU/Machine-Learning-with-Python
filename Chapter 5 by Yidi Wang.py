
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score as acc


# ## load iris data

# In[2]:


iris = datasets.load_iris()
iris.data


# In[3]:


iris.target


# ## convert the data into a dataframe with features f1-f4 and label columns

# In[6]:


col=['f1','f2','f3','f4']
df=pd.DataFrame(iris.data,columns=col)
df=pd.concat([df,pd.DataFrame(iris.target,columns=['label'])],axis=1)
df.head()


# ## scatter plot the data in the feature subspace spanned by f1 and f2, samples with different labels are colored differently

# In[7]:


get_ipython().run_line_magic('matplotlib', 'notebook')
for i in df.index:
#    print(df[['f1','f2']].loc[i].values,df.loc[i,'label'])
    if df.loc[i,'label']==0:
        color='red'
    elif df.loc[i,'label']==1:
        color='blue'
    else: color='yellow'
    plt.scatter(df.loc[i,'f1'],df.loc[i,'f2'],color=color)

plt.legend()
plt.show()


# ## formulate a classification problem and use KNeighborsClassifier to make predictions

# In[11]:


X_train,X_test,y_train,y_test=train_test_split(df[col],df['label'],test_size=0.3,random_state=123)


# In[12]:


# data normalization
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


# In[13]:


# training
knn=KNeighborsClassifier(n_neighbors=3,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)


# In[14]:


# predicting
y_pred=knn.predict(X_test_std)


# In[15]:


# evaluation by accuracy
acc(y_test,y_pred)


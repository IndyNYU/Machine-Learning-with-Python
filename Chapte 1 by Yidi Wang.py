
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# ## decision boundary comparison with synthetic data

# ## almost linearly separable data

# In[2]:


N=1000
ori = np.random.rand(N,2)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.scatter(ori[:,0],ori[:,1])


# In[5]:


label=np.array([0]*N)
df = pd.DataFrame({'A':ori[:,0],'B':ori[:,1]})
df['y']=label
df['y'][df['A']**2+0.25>df['B']]=1


# In[6]:


(df['y']==1).sum()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# here red dots stand for default samples, blue ones are non-default ones

plt.scatter(x=df[df['y']==1]['A'],
               y=df[df['y']==1]['B'], label=1, marker='o', color='red')
plt.scatter(x=df[df['y']==0]['A'],
               y=df[df['y']==0]['B'], label=0, marker='x', color = 'blue')
plt.xlabel('bill_amount1')
plt.ylabel('pay_amount1')


# In[8]:


# fea is a list of features to be used in the classification 
fea=['A','B']

sc = SC()
X_train = df[fea]
y_train = df['y']
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# training 
clf = SVC(C=1, kernel='rbf', class_weight='balanced')
clf.fit(X_train_std,y_train)

y_pred_train = clf.predict(X_train_std)

print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))


# In[9]:


from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)


# In[12]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plot_decision_regions(X_train.values, y_train.values, classifier=clf)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
plt.show()


# In[16]:


from sklearn.cross_validation import cross_val_score


# In[23]:


sklearn.__version__


# In[24]:


from sklearn.grid_search import GridSearchCV


# In[21]:


scores = cross_val_score(clf, X_train_std, y_train, cv=5, scoring='f1')


# In[22]:


scores.mean()


# In[65]:


# fea is a list of features to be used in the classification 
fea=['A','B']

sc = SC()
X_train = df[fea]
y_train = df['y']
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# training 
clf = SVC(C=1, kernel='linear', class_weight='balanced')
clf.fit(X_train_std,y_train)

y_pred_train = clf.predict(X_train_std)

print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))


# ## non-linearly separable yet quadratically separable data

# In[47]:


N=1000
ori = np.random.rand(N,2)
label=np.array([0]*N)
df = pd.DataFrame({'A':ori[:,0],'B':ori[:,1]})
df['y']=label
df['y'][(df['A']-0.5)**2+(df['B']-0.5)**2<0.17]=1


# In[48]:


(df['y']==1).sum()


# In[49]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# here red dots stand for default samples, blue ones are non-default ones

plt.scatter(x=df[df['y']==1]['A'],
               y=df[df['y']==1]['B'], label=1, marker='o', color='red')
plt.scatter(x=df[df['y']==0]['A'],
               y=df[df['y']==0]['B'], label=0, marker='x', color = 'blue')


# In[50]:


# fea is a list of features to be used in the classification 
fea=['A','B']

sc = SC()
X_train = df[fea]
y_train = df['y']
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# training 
clf = SVC(C=1, kernel='linear', class_weight='balanced')
clf.fit(X_train_std,y_train)

y_pred_train = clf.predict(X_train_std)

print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))


# In[51]:


# fea is a list of features to be used in the classification 
fea=['A','B']

sc = SC()
X_train = df[fea]
y_train = df['y']
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# training 
clf = SVC(C=1, kernel='rbf', class_weight='balanced')
clf.fit(X_train_std,y_train)

y_pred_train = clf.predict(X_train_std)

print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))


# # revisit credit data

# In[2]:


df = pd.read_csv('default.csv', index_col='ID')


# In[5]:


df.shape


# In[4]:


# fea is a list of features to be used in the classification 
fea=[]

# in this example,we 
for ele in df.columns:
    if 'PAY_' in ele:
        fea.append(ele)
        
train_balanced = df.sample(10000)
sc = SC()
X_train = train_balanced[fea]
y_train = train_balanced['default payment next month']
sc.fit(X_train)
X_train_std = sc.transform(X_train)

tic=time.time()
# training 
clf = SVC(C=1, kernel='linear', class_weight='balanced')
clf.fit(X_train_std,y_train)

y_pred_train = clf.predict(X_train_std)

print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))

toc=time.time()
print(toc-tic)


# In[6]:


for c in [0.001, 0.01, 0.1, 1, 10]:
    clf = SVC(C=c, kernel='linear', class_weight='balanced')
    clf.fit(X_train_std,y_train)
    y_pred_train = clf.predict(X_train_std)
    tic=time.time()
    print('----------------------------------------------------------')
    print('c=',c)
    print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
    print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
    print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
    print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))
    toc=time.time()
    print(toc-tic)


# ### runtime for proba=True

# In[9]:


train_balanced = df.sample(5000)
sc = SC()
X_train = train_balanced[fea]
y_train = train_balanced['default payment next month']
sc.fit(X_train)
X_train_std = sc.transform(X_train)

for c in [ 0.1, 1, 10]:
    tic=time.time()
    clf = SVC(C=c, kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X_train_std,y_train)
    y_pred_train = clf.predict(X_train_std)
    
    print('----------------------------------------------------------')
    print('c=',c)
    print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
    print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
    print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
    print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))
    toc=time.time()
    print(toc-tic)


# In[16]:


for c in [ 0.1, 1, 10]:
    tic=time.time()
    clf = SVC(C=c, kernel='linear', probability=True, tol=1e-1, max_iter=1E4, class_weight='balanced')
    clf.fit(X_train_std,y_train)
    y_pred_train = clf.predict(X_train_std)
    
    print('----------------------------------------------------------')
    print('c=',c)
    print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
    print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
    print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
    print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))
    toc=time.time()
    print(toc-tic)


# In[18]:


from sklearn.svm import LinearSVC as lsvc
clf1 = lsvc()

for c in [ 0.1, 1, 10]:
    tic=time.time()
    #clf1 = SVC(C=c, kernel='linear', probability=True, tol=1e-1, max_iter=1E4, class_weight='balanced')
    clf1.fit(X_train_std,y_train)
    y_pred_train = clf1.predict(X_train_std)
    
    print('----------------------------------------------------------')
    print('c=',c)
    print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
    print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
    print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
    print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))
    toc=time.time()
    print(toc-tic)


# In[10]:


for c in [0.1, 1, 10]:
    for gamma in [0.1, 1, 10]:
        tic=time.time()
        clf = SVC(C=c, kernel='rbf', gamma=gamma, probability=True, class_weight='balanced')
        clf.fit(X_train_std,y_train)
        y_pred_train = clf.predict(X_train_std)
        
        print('----------------------------------------------------------')
        print('c=',c, 'gamma',gamma)
        print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
        print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
        print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
        print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))
        toc=time.time()
        print('time:',toc-tic)


# In[15]:


for c in [0.1, 1, 10]:
    for gamma in [0.1, 1, 10]:
        tic=time.time()
        clf = SVC(C=c, kernel='rbf', gamma=gamma, probability=True, tol=1e-1, max_iter=1E4, class_weight='balanced')
        clf.fit(X_train_std,y_train)
        y_pred_train = clf.predict(X_train_std)
        
        print('----------------------------------------------------------')
        print('c=',c, 'gamma',gamma)
        print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
        print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
        print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
        print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))
        toc=time.time()
        print('time:',toc-tic)


# In[19]:


train_balanced = df.sample(5000)
sc = SC()
X_train = train_balanced[fea]
y_train = train_balanced['default payment next month']
sc.fit(X_train)
X_train_std = sc.transform(X_train)

for c in [0.1, 1, 10]:
    for gamma in [0.1, 1, 10]:
        tic=time.time()
        clf = SVC(C=c, kernel='rbf', gamma=gamma, probability=True, class_weight='balanced')
        clf.fit(X_train_std,y_train)
        y_pred_train = clf.predict(X_train_std)
        
        print('----------------------------------------------------------')
        print('c=',c, 'gamma',gamma)
        print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
        print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
        print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
        print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))
        toc=time.time()
        print('time:',toc-tic)


# In[7]:


for c in [0.001, 0.01, 0.1, 1, 10]:
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        tic=time.time()
        clf = SVC(C=c, kernel='rbf', gamma=gamma, class_weight='balanced')
        clf.fit(X_train_std,y_train)
        y_pred_train = clf.predict(X_train_std)
        
        print('----------------------------------------------------------')
        print('c=',c, 'gamma',gamma)
        print('acc -------', acc(y_true=y_train, y_pred=y_pred_train))
        print('precision:', precision_score(y_true=y_train, y_pred=y_pred_train))
        print('recall:', recall_score(y_true=y_train, y_pred=y_pred_train))
        print('f1:', f1_score(y_true=y_train, y_pred=y_pred_train))
        toc=time.time()
        print('time:',toc-tic)


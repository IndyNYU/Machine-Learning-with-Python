
# coding: utf-8

# In[1]:


'''
Using the MNIST dataset, which is a set of 70,000 small images of digits handwritten.
Scikit-Learn provides many helper functions to download popular datasets.
'''
# MNIST
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist


# In[2]:


X, y = mnist['data'], mnist['target']


# In[3]:


X.shape


# In[4]:


y.shape


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.axis("off")
plt.show()


# In[6]:


y[36000]


# In[7]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[8]:


import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[9]:


# Training a Binary Classifier.
# The "5-detector" will be an example of a binary classifier,
# capable of distinguishing between just two classes, 5 and not-5.
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# True for all 5s, False for all other digits.


# In[10]:


# Start with a Stochastic Gradient Descent Classifier.
# This classifier has the advantage of being capable of handling very large datasets.
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)


# In[11]:


# Performance Measures.
# Measuring Accuracy Using Cross-Validation.
# A good way to evaluate a model is to use cross-validation.
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits = 3, random_state = 42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


# In[12]:


# let's use the cross_val_score() function to evaluate this model.
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = "accuracy")


# In[13]:


# Let's look at a very dumb classifier that just classifies every single image in the not-5 class.
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y = None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype = bool)


# In[20]:


never_5_clf = Never5Classifier()


# In[21]:


cross_val_score(never_5_clf, X_train, y_train_5, cv = 3, scoring = "accuracy")


# In[22]:


# This demonstrates why accuracy is generally not the preferred performance measure for classifiers.
# A much better way to evaluate the performance of a classifier is to look at the confusion matrix.
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)


# In[24]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)


# In[26]:


confusion_matrix(y_train_5, y_train_5)


# In[28]:


# Precision and Recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)


# In[29]:


recall_score(y_train_5, y_train_pred)


# In[30]:


# Combine precision and recall into a single metric called the F1 score.
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


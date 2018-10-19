
# coding: utf-8

# Machine Larning Final Project by Yidi Wang
# 5/4/2018
# Forecast the S&P 500 Index with Machine Learning Classifiers
# Step 1. Data Preparing.
# Step 2. Data Seperating and Cleaning.
# Step 3. Fit Machine Learning Classifiers.
# Step 4. Performance Analysis.

# In[1]:


# Try to import packages I will use during the Process.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Step 1. Get the original data.

# In[2]:


# 1.1 Convert from CSV file to Pandas DataFrame.
# The data is downloaded from the Investing, convert the csv file to dataframe.
sp500 = pd.read_csv('D:/ML Data/S&P500.csv',index_col='Date')
Oil = pd.read_csv("D:/ML Data/CrudeOil.csv",index_col='Date')
Silver = pd.read_csv("D:/ML Data/Silver.csv",index_col='Date')
Gold = pd.read_csv("D:/ML Data/Gold.csv",index_col='Date')
VIX = pd.read_csv("D:/ML Data/CBOEVIX.csv",index_col='Date')
US2YearTNote = pd.read_csv("D:/ML Data/US2YearTNote.csv",index_col='Date')
US10YearTNote = pd.read_csv("D:/ML Data/US10YearTNote.csv",index_col='Date')
USBondYield = pd.read_csv("D:/ML Data/US10YearYield.csv",index_col='Date')
Euro50 = pd.read_csv("D:/ML Data/Euro50.csv",index_col='Date')
FTSE100 = pd.read_csv("D:/ML Data/FTSE100.csv",index_col='Date')
CAC40 = pd.read_csv("D:/ML Data/CAC40.csv",index_col='Date')
DAX = pd.read_csv("D:/ML Data/DAX.csv",index_col='Date')
HengSeng = pd.read_csv("D:/ML Data/HengSeng.csv",index_col='Date')
Nikkei225 = pd.read_csv("D:/ML Data/Nikkei225.csv",index_col='Date')
HS300 = pd.read_csv("D:/ML Data/Shanghai.csv",index_col='Date')


# In[3]:


# 1.2 Formulate the list of the dataframe.
list = [sp500, 
        Oil, Silver, Gold,
        VIX, US2YearTNote, US10YearTNote, USBondYield,
        Euro50, FTSE100, CAC40, DAX,
        HengSeng, HS300, Nikkei225]


# In[4]:


# 1.3 Reindex the dataframe according to the time order.
def reindex(item):
    return item.reindex(item.index[::-1])
list_total = []
for item in list:
    list_total.append(reindex(item))


# In[5]:


# 1.4 Keep the price and change of the price.
SP500_use = list_total[0]
SP500_use = SP500_use.rename(columns = {'Price':'SP500', 'Change %':'SP500Change'})

Oil_use = list_total[1]
Oil_use = Oil_use.rename(columns = {'Price':'Oil', 'Change %':'OilChange'})

Silver_use = list_total[2]
Silver_use = Silver_use.rename(columns = {'Price':'Silver', 'Change %':'SilverChange'})

Gold_use = list_total[3]
Gold_use = Gold_use.rename(columns = {'Price':'Gold', 'Change %':'GoldChange'})

VIX_use = list_total[4]
VIX_use = VIX_use.rename(columns = {'Price':'VIX', 'Change %':'VIXChange'})

US2YNote_use = list_total[5]
US2YNote_use = US2YNote_use.rename(columns = {'Price':'US2YNote', 'Change %':'US2YNoteChange'})

US10YNote_use = list_total[6]
US10YNote_use = US10YNote_use.rename(columns = {'Price':'US10YNote', 'Change %':'US10YNoteChange'})

US10YBond_use = list_total[7]
US10YBond_use = US10YBond_use.rename(columns = {'Price':'US10YBond', 'Change %':'US10YBondChange'})

Euro50_use = list_total[8]
Euro50_use = Euro50_use.rename(columns = {'Price':'Euro50', 'Change %':'Euro50Change'})

FTSE100_use = list_total[9]
FTSE100_use = FTSE100_use.rename(columns = {'Price':'FTSE100', 'Change %':'FTSE100Change'})

CAC40_use = list_total[10]
CAC40_use = CAC40_use.rename(columns = {'Price':'CAC40', 'Change %':'CAC40Change'})

DAX_use = list_total[11]
DAX_use = DAX_use.rename(columns = {'Price':'DAX', 'Change %':'DAXChange'})

HengSeng_use = list_total[12]
HengSeng_use = HengSeng_use.rename(columns = {'Price':'HengSeng', 'Change %':'HengSengChange'})

HS300_use = list_total[13]
HS300_use = HS300_use.rename(columns = {'Price':'HS300', 'Change %':'HS300Change'})

Nikkei225_use = list_total[14]
Nikkei225_use = Nikkei225_use.rename(columns = {'Price':'Nikkei225', 'Change %':'Nikkei225Change'})


# Step 2. Data Cleaning.

# In[6]:


# 2.1 Have an overview of the plot of the S&P 500 index from 2010 to 2018.
SP500_use['SP500'].plot(figsize=(8,5),legend=True,title='The SP500 Index',color='black')


# In[7]:


# 2.2 Have an overview of the plot of the change of S&P 500 index from 2010 to 2018.
SP500_use['SP500Change'].plot(figsize=(8,5),legend=True,title='The SP500 Index',color='black')


# In[8]:


# 2.3 Merge the dataframe to get a whole dataframe to use.
list_use = [SP500_use, 
            Oil_use, Silver_use, Gold_use,
            VIX_use, US2YNote_use, US10YNote_use, US10YBond_use,
            Euro50_use, FTSE100_use, CAC40_use, DAX_use,
            HengSeng_use, HS300_use, Nikkei225_use]


# In[9]:


total_data = pd.concat(list_use,axis=1, join='inner',join_axes=[list_use[0].index])


# In[10]:


# 2.4 Only keep the price and the change of the price.
pricechange_data = total_data[['SP500Change','OilChange','SilverChange','GoldChange','VIXChange','US2YNoteChange','US10YNoteChange','US10YBondChange','Euro50Change','FTSE100Change','CAC40Change','DAXChange','HengSengChange','HS300Change','Nikkei225Change']]


# In[11]:


price_data = total_data[['SP500','Oil','Silver','Gold','VIX','US2YNote','US10YNote','US10YBond','Euro50','FTSE100','CAC40','DAX','HengSeng','HS300','Nikkei225']]


# In[12]:


# 2.5 Try to get the correlation of the variables and order the correlation.
price_data.corr()['SP500'].sort_values(ascending=False)


# Step 3. Get the training data and test data.

# In[13]:


# 3.1 Get the training data and test data.
pricechange_data['SP500Change']=np.where(pricechange_data['SP500Change']>0,1,pricechange_data['SP500Change'])
pricechange_data['SP500Change']=np.where(pricechange_data['SP500Change']<0,-1,pricechange_data['SP500Change'])


# In[14]:


# 3.2 Look at the data.
pricechange_data['SP500Change'].head(5)
pricechange_data['SP500Change'].tail(5)


# In[15]:


# 3.3 Because the data is enough, so I decide to drop the na data.
total_data = pricechange_data.dropna()


# In[16]:


# 3.4 Specific information training and testing set.
total_data.info()
train_data = total_data.loc['15-Aug-11':'29-Dec-17']
test_data = total_data.loc['4-Jan-18':'29-Mar-18']


# In[17]:


# 3.5 Have a overview of the dataset and try to make the ratio of traing to test balanced.
train_data.info()
test_data.info()


# In[18]:


# 3.6 Seperate the variable parameter matrix and the label vector.
# In other words, the X is the variable matrix, the y is the label vector.
X_train = train_data[['OilChange','SilverChange','GoldChange','VIXChange','US2YNoteChange','US10YNoteChange','US10YBondChange','Euro50Change','FTSE100Change','CAC40Change','DAXChange','HengSengChange','HS300Change','Nikkei225Change']]
y_train = train_data['SP500Change']
y_train = pd.DataFrame(y_train)
X_test = test_data[['OilChange','SilverChange','GoldChange','VIXChange','US2YNoteChange','US10YNoteChange','US10YBondChange','Euro50Change','FTSE100Change','CAC40Change','DAXChange','HengSengChange','HS300Change','Nikkei225Change']]
y_test = test_data['SP500Change']
y_test = pd.DataFrame(y_test)


# Step 4. Fit Machine Learning Classifiers.

# In[19]:


# 4.1 Select performance and define a function.
# As this is the classification problem, 
# so i will use the precision and accuracy as performance index.
from sklearn.metrics import precision_score, accuracy_score
def predict(model):
    model_use = model.fit(X_train, y_train)
    predict = model_use.predict(X_test)
    return predict
def performance(predict):
    predict = pd.DataFrame(predict, index = y_test.index)
    precision = precision_score(y_test, predict)
    accuracy = accuracy_score(y_test, predict)
    return[precision, accuracy]


# In[20]:


# 4.2 Train the model and get the performance.
# The analysis of KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
KNN_classifiers = [KNeighborsClassifier(1),KNeighborsClassifier(2),KNeighborsClassifier(3),
                   KNeighborsClassifier(4),KNeighborsClassifier(5),KNeighborsClassifier(6),
                   KNeighborsClassifier(7),KNeighborsClassifier(8),KNeighborsClassifier(9)]
list_KNN = []
for classifier in KNN_classifiers:
    list_KNN.append(performance(predict(classifier)))


# In[21]:


list_KNN


# In[22]:


plot(list_KNN)


# In[23]:


# 4.3 The analysis of the Logistic Regression.
from sklearn.linear_model import LogisticRegression
LR_l1_1 = LogisticRegression(penalty='l1',C=0.1)
LR_l1_2 = LogisticRegression(penalty='l1',C=0.3)
LR_l1_3 = LogisticRegression(penalty='l1',C=0.5)
LR_l1_4 = LogisticRegression(penalty='l1',C=0.8)
LR_l1_5 = LogisticRegression(penalty='l1',C=1.0)

LR_l2_1 = LogisticRegression(penalty='l2',C=0.1)
LR_l2_2 = LogisticRegression(penalty='l2',C=0.3)
LR_l2_3 = LogisticRegression(penalty='l2',C=0.5)
LR_l2_4 = LogisticRegression(penalty='l2',C=0.8)
LR_l2_5 = LogisticRegression(penalty='l2',C=1.0)

LogisticRegression_Classifiers = [LR_l1_1,LR_l1_2,LR_l1_3,LR_l1_4,LR_l1_5,
                                  LR_l2_1,LR_l2_2,LR_l2_3,LR_l2_4,LR_l2_5,]
list_LR = [ ]
for classifier in LogisticRegression_Classifiers:
    list_LR.append(performance(predict(classifier)))


# In[24]:


list_LR


# In[25]:


plot(list_LR)


# In[26]:


# 4.4 The analysis of SVC.
from sklearn.svm import SVC
SVC_rbf = SVC(kernel='rbf')
SVC_linear = SVC(kernel='linear')
SVC_poly = SVC(kernel='poly')

SVC_Classifiers= [SVC_rbf, SVC_linear, SVC_poly]
list_SVC = [ ]
for classifier in SVC_Classifiers:
    list_SVC.append(performance(predict(classifier)))


# In[27]:


list_SVC


# In[28]:


SVC_01 = SVC(kernel='linear',C=0.1)
SVC_03 = SVC(kernel='linear',C=0.3)
SVC_05 = SVC(kernel='linear',C=0.5)
SVC_08 = SVC(kernel='linear',C=0.8)
SVC_10 = SVC(kernel='linear',C=1.0)
SVC_15 = SVC(kernel='linear',C=1.5)
SVC_20 = SVC(kernel='linear',C=2.0)
SVC_n_Classifiers = [SVC_01, SVC_03, SVC_05, SVC_08, SVC_10, SVC_15, SVC_20]

list_n_SVC = [ ]
for classifier in SVC_n_Classifiers:
    list_n_SVC.append(performance(predict(classifier)))


# In[29]:


list_n_SVC


# In[30]:


plot(list_n_SVC)


# In[31]:


# 4.5 The analysis of decision tree.


# In[32]:


from sklearn.tree import DecisionTreeClassifier
DT_1 = DecisionTreeClassifier(max_depth=1)
DT_2 = DecisionTreeClassifier(max_depth=2)
DT_3 = DecisionTreeClassifier(max_depth=3)
DT_4 = DecisionTreeClassifier(max_depth=4)
DT_5 = DecisionTreeClassifier(max_depth=5)
DT_6 = DecisionTreeClassifier(max_depth=6)
DT_7 = DecisionTreeClassifier(max_depth=7)
DT_8 = DecisionTreeClassifier(max_depth=8)


# In[33]:


DT_Classifiers = [DT_1, DT_2, DT_3, DT_4, DT_5, DT_6, DT_7, DT_8]
list_DT = []
for classifier in DT_Classifiers:
    list_DT.append(performance(predict(classifier)))


# In[34]:


list_DT


# In[35]:


plot(list_DT)


# In[36]:


# 4.5 The analysis of Neural Networks.
from sklearn.neural_network import MLPClassifier


# In[37]:


NN_identity = MLPClassifier(activation='identity')
NN_logistic = MLPClassifier(activation='logistic')
NN_tanh = MLPClassifier(activation='tanh')
NN_relu = MLPClassifier(activation='relu')
NN_Classifiers = [NN_identity, NN_logistic, NN_tanh, NN_relu]
list_NN_activation = []
for classifier in NN_Classifiers:
    list_NN_activation.append(performance(predict(classifier)))


# In[38]:


list_NN_activation


# In[39]:


plot(list_NN_activation)


# In[40]:


# 4.6 The analysis of AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
performance(predict(AdaBoostClassifier()))


# Step 5. Performance and Time Analysis.

# In[41]:


get_ipython().run_cell_magic('time', '', 'performance(predict(KNeighborsClassifier(4)))')


# In[42]:


get_ipython().run_cell_magic('time', '', "performance(predict(LogisticRegression(penalty='l2',C=0.8)))")


# In[43]:


get_ipython().run_cell_magic('time', '', "performance(predict(SVC(kernel='linear',C=0.8)))")


# In[44]:


get_ipython().run_cell_magic('time', '', 'performance(predict(DecisionTreeClassifier(max_depth=2)))')


# In[45]:


get_ipython().run_cell_magic('time', '', "performance(predict(MLPClassifier(activation='logistic')))")


# In[46]:


get_ipython().run_cell_magic('time', '', 'performance(predict(AdaBoostClassifier()))')


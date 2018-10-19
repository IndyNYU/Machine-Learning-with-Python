
# coding: utf-8

# In[ ]:


# Hands-on Machine Learning with Scikit-Learn & TensorFlow
# Chapter 2 by Yidi Wang


# In[1]:


'''
Build a model of housing prices in California using the California census data.
This data has metrics such as the population, median income, median housing price,
and so on for each block group in California.
Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data.
This model will learn from these data and be able to predict the median housing price in any district,
given all the other metrics.
'''


# In[ ]:


# 1. Frame the Problem.
# The first question is the business objective and the whole pipeline.
# The next question is what is the current solution and the reference performance.
# Start to design the system.
# It is clearly a typical supervised learning task since we have the labeled training examples.
# Moreover, it is also a typical regression task, since we are asked to predict a value.
# More specifically, this is a multivariate regression problem since the system will use multiple features to make a prediction.
# Finally, there is no continuous flow of data rapidly, and the data is small enough to fit in memory.


# In[ ]:


# 2. Select a Performance Measure.
# A typical performance measure for regression problem is the Root Mean Square Error.
# Even though the RMSE is generally the preferred performance measure for regression tasks,
# in some contexts you may prefer to use another function.
# The higher the norm index, the more it focuses on large values and neglects small ones.
# This is why the RMSE is more sensitive to outliers than the MAE.


# In[ ]:


# 3. Check the Assumptions.
# Consult with the downstream system about the output of the machine learning algorithm.
# It is confirmed with the numerical value.


# In[47]:


# 4. Get the data.
# Download the comma-separted value(CSV) file.
import pandas as pd
housing  = pd.read_csv('E:\MLdata\handson-ml-master\handson-ml-master\datasets\housing\housing.csv')


# In[48]:


# 5. Take a Quick Look at the Data Structure.
# Let's take a look at the top five rows using the DataFrame's head() method.
housing.head()


# In[49]:


# Each row represents one district.
# There are 10 attributes.
# The info() method is useful to get a quick description of the data.
housing.info()


# In[50]:


# There are 20,640 instances in the dataset, which means that it is fairly small by Machine Learning standards.
# Notice that the total_bedrooms attribute has only 20,433 non-null valuse, meaning that 207 districts are missing this feature.
# All attributes are numerical, except the ocean_proximity field. Its type is object.
# Find out what categories exist and how many districts belong to each category by using the value_counts() method.
housing['ocean_proximity'].value_counts()


# In[51]:


# Let's look at the other fields. The describe() method shows a summary of the numerical attributes.
housing.describe()


# In[52]:


# The null values are ignored.
# Another quick way to get a fell of the type of data is to plot a histogram for each numerical attribute.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize = (20, 15))
plt.show()


# In[ ]:


# Notice a few things in these histograms.
# 1. First, the median income attribute does not look like it is expressed in US dollars, it is capped in [0.5, 15].
# 2. The housing median age and the median house value were also capped.
#    The latter may be a serious problem since it is the target attribute.
# 3. These attributes have very different scales.
# 4. Finally, the histograms are tail heavy: they extend much farther to the right of the median than to the left.
#    So, we will transform these attributes to have more bell-shaped distributions.


# In[53]:


# 6. Create a Test Set.
# Creating a test set is theoretically quite simple: just pick some instances randomly, 20% of the dataset.
# Scikit-Learn provides a few functions to split datasets into multiple subsets in various ways.
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[54]:


# Well, we only consider purely random sampling methods.
# It may introduce a significant sampling bias. 
# Consider about the stratified sampling. Suppose that the median income is a very important attribute.
# To ensure that the test set is representative of the various categories of incomes in the whole dataset.
# We should not have too many strata, and each stratum should be large enough.
# To create an income category attribute by dividing the median income by 1.5, and rouding up using ceil and merging all.
import numpy as np
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace = True)


# In[55]:


# To do stratified sampling based on the income category.
# For this, we can use Scikit-Learn's StratefiedShuffleSplit class.
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[56]:


# Let's see if this worked as expected.
housing['income_cat'].value_counts() / len(housing)


# In[57]:


# Now we should remove the income_cat attribute so the data is back to its original state.
for set in (strat_train_set, strat_test_set):
    set.drop(['income_cat'], axis = 1, inplace = True)


# In[58]:


# 7. Discover and Visualize the Data to Gain Insighhs.
# First, make sure we have put the test set aside and only exploring the training set.
housing = strat_train_set.copy()


# In[59]:


# 8. Visualizing Geographical Data.
# Since there is geographical information, it is a good idea to create a scatterplot of all districts to visualize the data.
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', title = 'A geographical scatterplot of the data.')


# In[60]:


# Set the alpha option to 0.1 makes it much easier to visualize the places where there is a high density of data points.
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', 
             title = 'A better visualization highlighting high-density areas.', alpha = 0.1)


# In[61]:


# Need to play around with visualization parameters to make the patterns stand out.
# Let's look at the housing prices. 
# The radius of each circle represents the district's population, and the color represents the price.
# We will use a predefined color map(option cmap) called jet, which ranges from blue to red.
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.4,
             s =  housing['population'] / 100, label = 'population', c = 'median_house_value',
             cmap = plt.get_cmap('jet'), colorbar = True, title = 'California housing prices.')
plt.legend()


# In[62]:


# The image tells you that the housing prices are very much related to the location and population density.
# It will probably be useful to use a clustering algorithm to detect the main clusters,
# and add new features that measure the proximity to the cluster centers.
# 9. looking for Correlations.
# Since the dataset is not too large, you can easily compute the standard correlation coefficient between each pairs.
corr_matrix = housing.corr()
# Look at how much each attribute correlates with the median house value.
corr_matrix['median_house_value'].sort_values(ascending = False)


# In[63]:


# The correlation coefficient ranges from -1 to +1.
# Coefficients close to zero mean that there is no linear correlation.
# Another way to check for correlation between attributes is to use Pandas' sactter_matrix function.
from pandas.tools.plotting import scatter_matrix
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize = (12, 8))


# In[64]:


# The most promising attribute to predict the median house value is the median income.
housing.plot(kind = 'scatter', x = 'median_income', y = 'median_house_value', alpha = 0.1,
             title = 'Meidan income versus median house value.')


# In[65]:


# This plot reveals a few things.
# First, the correlation is indeed very strong, we can clearly see the upward trend.
# Second, the price cap is clear to see as $50,000.


# In[66]:


# 9. Experimengting with Attribute Combinations.
# Identify a few data quirks and want to clean them up, correlation with attributes and targets.
# Also, some attributes have a tail-heavy distribution.
# One thing to do before actually preparing the data for Machine Learning is to try out various attribute combinations.
# Create some new attributes.
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']


# In[67]:


# Let's look at the correlation matrix again.
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)


# In[68]:


# 10. Prepare the Data for Machine Learning Algorithms.
# It's time to prepare the data for ML algorithms. Write functions to do that.
# First, let's revert to a clean training set and seperate the predictors and the labels.
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()


# In[69]:


# 11. Data Cleaning.
# Most Machine Learning algorithms cannot work with missing features.
# Notice that the total_bedrooms attribute has some missing values, so let's fix it.
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
# Since median could only be calculated for the numerical values, so we drop the text attribute.
housing_num = housing.drop('ocean_proximity', axis = 1)
imputer.fit(housing_num)


# In[70]:


imputer.statistics_


# In[71]:


housing_num.median().values


# In[72]:


# Use the trained imputer to transform the training set by replacing missing values by the learned median.
X = imputer.transform(housing_num)


# In[73]:


# The result is a plain Numpy array. Put it back to a Pandas DataFrame.
housing_tr = pd.DataFrame(X, columns = housing_num.columns)


# In[74]:


# 12. Handling Text and Categorical Attributes.
# Scikit-Learn provides a transformer for this task called LabelEncoder.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


# In[75]:


print(encoder.classes_)


# In[76]:


# Scikit-Learn provides a OneHotEncoder encoder to convert integer categorical values into one-hot vectors.
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
housing_cat_1hot


# In[77]:


# 13. Custom Transformers.
# Although Scikit-Learn provides many useful transformers, you will need to write your own for tasks.
# Here is a samll transformer class that adds the combined attributes we discussed earlier.
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix , bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attibutes = attr_adder.transform(housing.values)


# In[78]:


# 14. Feature Scaling
# One of the most important transformations you need to apply to your data is feature scaling.
# Note that sacling the target values is generally not required.
# There are two common ways to get all attributes to have the same scale: min-max and standardization.


# In[79]:


# 15. Transformation Pipelines
# There are many data transformation steps that need to be executed in the right order.
# Scikit-Learn provides the Pipeline class to help with such sequences of transformations.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', Imputer(strategy = 'median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])
housing_num_tr = num_pipeline.fit_transform(housing_num)


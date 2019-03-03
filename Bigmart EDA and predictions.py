
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


train = pd.read_csv('Train_BigMart.csv')


# In[ ]:


train['Item_Fat_Content'].unique()


# In[ ]:


train['Outlet_Establishment_Year'].unique()


# In[ ]:


train['Outlet_Size'].unique()


# In[ ]:


train.describe()


# In[ ]:


train['Item_Visibility'].hist(bins=20)


# In[ ]:


train['Item_Fat_Content'].value_counts()


# In[ ]:


train['Outlet_Size'].value_counts()


# In[ ]:


train.boxplot(column='Item_MRP', by='Outlet_Size')


# In[ ]:


train.boxplot(column='Item_Visibility', by='Outlet_Type')


# In[ ]:



train['Outlet_Size'].mode()[0]


# In[ ]:



# fill the na for outlet size with medium
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])


# In[ ]:



# fill the na for item weight with the mean of weights
train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].mean())


# In[ ]:


train.boxplot(column='Item_Visibility')


# In[ ]:


# delete the observations

Q1 = train['Item_Visibility'].quantile(0.25)
Q3 = train['Item_Visibility'].quantile(0.75)
IQR = Q3 - Q1
filt_train = train.query('(@Q1 - 1.5 * @IQR) <= Item_Visibility <= (@Q3 + 1.5 * @IQR)')


# In[ ]:


filt_train.shape, train.shape


# In[ ]:


train = filt_train
train.shape


# In[ ]:


train['Item_Visibility_bins'] = pd.cut(train['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])


# In[ ]:


train['Item_Visibility_bins'] = train['Item_Visibility_bins'].replace(NaN, 'Low Viz')


# In[ ]:


train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')


# In[ ]:


train['Item_Fat_Content'] = train['Item_Fat_Content'].replace('reg', 'Regular')


# In[ ]:


#choosing the Fat content, item vizibility bins, outlet size, loc type and type for LABEL ENCODER
le = LabelEncoder()


# In[ ]:


train['Item_Fat_Content'].unique()


# In[ ]:


train['Item_Fat_Content'] = le.fit_transform(train['Item_Fat_Content'])


# In[ ]:


train['Item_Visibility_bins'] = le.fit_transform(train['Item_Visibility_bins'])


# In[ ]:


train['Outlet_Size'] = le.fit_transform(train['Outlet_Size'])


# In[ ]:


train['Outlet_Location_Type'] = le.fit_transform(train['Outlet_Location_Type'])


# In[ ]:


# create dummies for outlet type
dummy = pd.get_dummies(train['Outlet_Type'])
dummy.head()


# In[ ]:


train = pd.concat([train, dummy], axis=1)


# In[ ]:


# in linear regression that correlated features should not be present

train.corr()[((train.corr() < -0.85) | (train.corr() > 0.85)) & (train.corr() != 1)]


# In[ ]:


train.dtypes


# In[ ]:


# got to drop all the object types features
train = train.drop(['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type'], axis=1)


# In[ ]:


train.columns


# In[ ]:


# build the linear regression model
X = train.drop('Item_Outlet_Sales', axis=1)
y = train.Item_Outlet_Sales


# In[ ]:


test = pd.read_csv('Test_BigMart.csv')
test['Outlet_Size'] = test['Outlet_Size'].fillna('Medium')


# In[ ]:


test['Item_Visibility_bins'] = pd.cut(test['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])


# In[ ]:


test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].mean())


# In[ ]:


test['Item_Visibility_bins'] = test['Item_Visibility_bins'].replace(NaN, 'Low Viz')
test['Item_Visibility_bins'].head()


# In[ ]:


test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace('reg', 'Regular')


# In[ ]:


test['Item_Fat_Content'] = le.fit_transform(test['Item_Fat_Content'])


# In[ ]:


test['Item_Visibility_bins'] = le.fit_transform(test['Item_Visibility_bins'])


# In[ ]:


test['Outlet_Size'] = le.fit_transform(test['Outlet_Size'])


# In[ ]:


test['Outlet_Location_Type'] = le.fit_transform(test['Outlet_Location_Type'])


# In[ ]:


dummy = pd.get_dummies(test['Outlet_Type'])
test = pd.concat([test, dummy], axis=1)


# In[ ]:


X_test = test.drop(['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type'], axis=1)


# In[ ]:


X.columns, X_test.columns


# In[ ]:


lin = LinearRegression()


# In[ ]:


lin.fit(X, y)
predictions = lin.predict(X_test)


# In[ ]:


# # create submission file
# submission = pd.DataFrame(data=[], columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
# submission['Item_Identifier'] = test['Item_Identifier']
# submission['Outlet_Identifier'] = test['Outlet_Identifier']
# submission['Item_Outlet_Sales'] = predictions
# submission.to_csv('submission.csv', index=False)
# submission.head()


# In[ ]:


# my first score was 1203 points - Linear regression only


# In[ ]:


# decision tree
dtree_class = DecisionTreeClassifier(criterion='gini', max_depth=25)
y = y.astype(int)


# In[ ]:


dtree_class.fit(X, y)


# In[ ]:


accuracy_score(y, dtree_class.predict(X))


# In[ ]:


r2_score(y, dtree_class.predict(X))


# In[ ]:


pred = dtree_class.predict(X_test)
pred


# In[ ]:


# # create submission file
# submission = pd.DataFrame(data=[], columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
# submission['Item_Identifier'] = test['Item_Identifier']
# submission['Outlet_Identifier'] = test['Outlet_Identifier']
# submission['Item_Outlet_Sales'] = pred
# submission.to_csv('submission.csv', index=False)
# submission.head()


# In[ ]:


# score was 1712 points - Decision Tree Classifier!!!


# In[ ]:


dtree_reg = DecisionTreeRegressor(criterion='mse', max_depth=10)


# In[ ]:


dtree_reg.fit(X, y)


# In[ ]:


pred = dtree_reg.predict(X_test)
pred


# In[ ]:


# # create submission file
# submission = pd.DataFrame(data=[], columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
# submission['Item_Identifier'] = test['Item_Identifier']
# submission['Outlet_Identifier'] = test['Outlet_Identifier']
# submission['Item_Outlet_Sales'] = pred
# submission.to_csv('submission2.csv', index=False)
# submission.head()


# In[ ]:


# score was 1289 points - Decision Tree Regression!!!


# In[ ]:


cross_val_score(lin, X, y, cv=5, scoring='r2')


# In[ ]:


cross_val_score(dtree_reg, X, y, cv=5, scoring='r2')


# In[ ]:


# cross_val_score(dtree_class, X, y, cv=5, scoring='roc_auc') - results in an error


# In[ ]:


r2_score(y, lin.predict(X))


# In[ ]:


r2_score(y, dtree_reg.predict(X))


# In[ ]:


avg_pred = (lin.predict(X) + dtree_reg.predict(X)) / 2


# In[ ]:


r2_score(y, avg_pred)


# In[ ]:


wavg_pred = lin.predict(X)*0.1 + dtree_reg.predict(X)*0.9


# In[ ]:


r2_score(y, wavg_pred)


# In[ ]:


rmf = RandomForestClassifier(n_estimators=100, max_depth=10)


# In[ ]:


rmf.fit(X, y)


# In[ ]:


r2_score(y, rmf.predict(X))


# In[ ]:


accuracy_score(y, rmf.predict(X))


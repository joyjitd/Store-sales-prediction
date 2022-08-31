#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries to import
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import  r2_score
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


# Dataset load
df_train = pd.read_csv("E:/PROJECTS/Machine Learning/9. Big Mart Sales Prediction/Dataset/train.csv")
df_test = pd.read_csv("E:/PROJECTS/Machine Learning/9. Big Mart Sales Prediction/Dataset/test.csv")


# In[3]:


print(df_train.shape)
df_train.head(5)


# In[4]:


print(df_test.shape)
df_test.head(5)


# Since the test dataset do not contain any target output(Item_Outlet_Sales) so we can not use the test dataset.Rather we will split the train datatset into test and split later formodel building.

# In[5]:


# Train dataset load
df = df_train.copy()


# In[6]:


# Datafile exploration
df.info()


# In[7]:


df.head(4)


# In[8]:


df.describe()


# ### Exploratory Data Analysis

# In[9]:


# Null value check
df.isnull().sum()


# In[10]:


# Dropping the outlet_size missing values
df.dropna(subset = ['Outlet_Size'], inplace = True)


# In[11]:


# Imputing the Item_weight missing values with central tendencies
df['Item_Weight'].fillna(df.Item_Weight.median(),inplace=True)


# In[12]:


df.isnull().sum()


# In[13]:


# Item_Visibility column
df[df['Item_Visibility']==0]['Item_Visibility'].count()


# In[14]:


df['Item_Visibility'].fillna(df['Item_Visibility'].median(), inplace=True)


# In[15]:


# Outlet year
df['Outlet_Establishment_Year'].value_counts()


# In[16]:


# We will convert this years into age of the outlets 
df['Outlet_Years'] = 2009-df['Outlet_Establishment_Year']
df['Outlet_Years'].describe()


# In[17]:


#Removing the outlet establishment column
df.drop("Outlet_Establishment_Year", axis = 1, inplace = True)


# In[18]:


# Item identifier
df['Item_Identifier'].value_counts()


# In[19]:


#Lets group the 16 categories into main 3 groups
df['New_Item_type'] = df['Item_Identifier'].apply(lambda x: x[0:2])
df['New_Item_type'] = df['New_Item_type'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
df['New_Item_type'].value_counts()


# In[20]:


# Checking the distribution of the variables
df.hist()
plt.show()


# In[21]:


# Item_Type value count
plt.figure(figsize=(15,10))
sns.countplot(df.Item_Type)
plt.xticks(rotation=90)
print(df.Item_Type.value_counts())
plt.show()
warnings.filterwarnings("ignore")


# In[22]:


# Outlet_size ditribution
plt.figure(figsize=(10,8))
sns.countplot(df.Outlet_Size)
print(df.Outlet_Size.value_counts())
plt.show()


# In[23]:


# Outlet location type
plt.figure(figsize=(10,8))
sns.countplot(df.Outlet_Location_Type)
print(df.Outlet_Location_Type.value_counts())
plt.show()


# In[24]:


# Outlet_type distributionplt.figure(figsize=(10,8))
plt.figure(figsize=(10,8))
sns.countplot(df.Outlet_Type)
plt.xticks(rotation=10)

print(df.Outlet_Type.value_counts())


# In[25]:


# Item weight and item outlet sales analysis
plt.figure(figsize=(13,9))
plt.xlabel('Item_Weight')
plt.ylabel('Item_Outlet_Sales')
plt.title('Item_Weight and Item_Outlet_Sales Analysis')
sns.scatterplot(x='Item_Weight', y='Item_Outlet_Sales',hue='Item_Type', size='Item_Weight',data=df)
plt.show()


# In[26]:


# Item visibility and maximum retail price
plt.figure(figsize=(12,7))
plt.xlabel('Item_Visibility')
plt.ylabel('Maximum Retail Price')
plt.title('Item_Visibility and Maximum Retail Price')
plt.plot(df.Item_Visibility, df.Item_MRP, ".",alpha = 0.3)
plt.show()


# In[27]:


# Impact of outlet_type on item_outlet_sales
Item_Type_pivot = df.pivot_table(index='Outlet_Type', values="Item_Outlet_Sales", aggfunc=np.median)

Item_Type_pivot.plot(kind='bar',color='brown',figsize=(12,7))
plt.xlabel('Outlet_Type')
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[28]:


# Item of item_fat content on item_outlet_sales
Item_Type_pivot = df.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)

Item_Type_pivot.plot(kind='bar',color='blue',figsize=(12,7))
plt.xlabel('Item_Fat_Content')
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
print(df['Item_Fat_Content'].value_counts())
plt.show()


# In[29]:


# Renaming the same types in item_fat_content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat',})


# In[30]:


Item_Type_pivot = df.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)

Item_Type_pivot.plot(kind='bar',color='blue',figsize=(12,7))
plt.xlabel('Item_Fat_Content')
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
print(df['Item_Fat_Content'].value_counts())
plt.show()


# In[31]:


# Checking the correlation between the variables
plt.figure(figsize=(35,15))
sns.heatmap(df.corr(),vmax=1, square=True,annot=True, cmap='viridis')
plt.title('Correlation between different attributes')
plt.show()


# In[32]:


df.head(5)


# ### FeatureTransformation

# In[33]:


# Dropping unnecessary columns
df.drop(["Item_Identifier","Outlet_Identifier","Item_Visibility","New_Item_type","Outlet_Type","New_Item_type"], axis = 1, inplace = True)


# In[34]:


df.head()


# In[35]:


label = LabelEncoder()
df.Item_Fat_Content = label.fit_transform(df.Item_Fat_Content)
df.Item_Type = label.fit_transform(df.Item_Type)
df.Outlet_Location_Type = label.fit_transform(df.Outlet_Location_Type)
df.Outlet_Size = label.fit_transform(df.Outlet_Size)


# In[36]:


df.head(4)


# In[37]:


# Dependent and independent variable separation
x = df.drop(columns = "Item_Outlet_Sales")
y = df.Item_Outlet_Sales


# In[38]:


x.head(4)


# In[39]:


y


# In[40]:


# Multicollinearity check

## VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                          for i in range(len(x.columns))]
  
print(vif_data)


# Thus the dataset needs no-scaling since the vif is low for all the variables

# In[41]:


# Train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


# ### Linear Regression

# In[42]:


model_lr = LinearRegression()
model_lr.fit(x_train, y_train)

y_pred = model_lr.predict(x_test)
print(r2_score(y_test, y_pred))

print(f"The model prediction on train dataset: {round(model_lr.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_lr.score(x_test, y_test),2)}")


# ### Support Vector Regressor

# In[43]:


model_svr = SVR()
model_svr.fit(x_train , y_train)
print(f"The model prediction on train dataset: {round(model_svr.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_svr.score(x_test, y_test),2)}")


# ### KNN

# In[44]:


model_knn = KNeighborsRegressor()
model_knn.fit(x_train , y_train)
print(f"The model prediction on train dataset: {round(model_knn.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_knn.score(x_test, y_test),2)}")


# ### Decision Tree

# In[45]:


model_dt = DecisionTreeRegressor()
model_dt.fit(x_train , y_train)
print(f"The model prediction on train dataset: {round(model_dt.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_dt.score(x_test, y_test),2)}")


# ### Random Forest

# In[46]:


model_rf = RandomForestRegressor()
model_rf.fit(x_train , y_train)
print(f"The model prediction on train dataset: {round(model_rf.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_rf.score(x_test, y_test),2)}")


# ### Extra Tree Regressor

# In[47]:


model_et = ExtraTreesRegressor()
model_et.fit(x_train , y_train)
print(f"The model prediction on train dataset: {round(model_et.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_et.score(x_test, y_test),2)}")


# ### Adaptive Boost Regressor

# In[48]:


model_ada = AdaBoostRegressor(base_estimator=model_dt)
model_ada.fit(x_train , y_train)
print(f"The model prediction on train dataset: {round(model_ada.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_ada.score(x_test, y_test),2)}")


# ### Gradient Boosting Regressor

# In[49]:


model_gbo = GradientBoostingRegressor()
model_gbo.fit(x_train , y_train)
print(f"The model prediction on train dataset: {round(model_gbo.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_gbo.score(x_test, y_test),2)}")


# Thus we can see that Gradient boosting model gives us the highest sore of 60% on the train datatset.We will use this model for final deployment.

# In[50]:


## BEST MODEL SAVE
pickle.dump(model_gbo, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
# %matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style


# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


from datetime import datetime, date, time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, average_precision_score,classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer,KNNImputer
from scipy.stats import norm
from scipy import stats


# In[217]:


df=pd.read_csv('E:/data/data science/data course/DATASET/winequalityN.csv')


# In[218]:


df.shape


# In[219]:


df = df.drop_duplicates(keep='first').reset_index(drop=True)


# In[220]:


df.shape


# In[221]:


df.info()


# In[222]:


df.head()


# In[223]:


df.tail()


# In[224]:


df.describe()


# In[225]:


print("\n\nRows     : ", df.shape[0])
print("\nColumns  : ", df.shape[1])
print("\nFeatures : \n", df.columns.tolist())
print("\nMissing values :  ", df.isnull().sum().values.sum())
print("\nUnique values :  \n", df.nunique())


# In[226]:


df['type']=df['type'].apply(lambda x : 1 if x == 'white' else 2 )
print('unique type:', df['type'].nunique())


# In[227]:


df.isnull().sum()


# In[228]:


imputer=KNNImputer()
df=pd.DataFrame(imputer.fit_transform(df),columns=df.columns)


# In[229]:


df['quality level'] = df['quality'].apply( lambda x: 'Low' if x <= 4 else 'Medium' if  x <= 6 else 'High' )
print('unique quality level:', df['quality level'].nunique())


# In[230]:


df['quality'].value_counts().plot(kind='bar');


# In[231]:


df.groupby(['type','quality'])['quality'].count()


# In[232]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df[['type','quality']].groupby(['type']).mean().plot.bar(ax=ax[0])
ax[0].set_title('type vs quality')
sns.countplot('type',hue='quality',data=df,ax=ax[1])
ax[1].set_title('type vs quality')
plt.show()


# In[233]:


corrlation = df.corr()
corrlation['quality'].sort_values(axis=0, ascending=False)


# In[234]:


corrlation['type'].sort_values(axis=0, ascending=False)


# In[235]:


plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(),annot=True, cmap='RdYlGn');


# In[236]:


f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(28,5))

sns.scatterplot(x=df['alcohol'], y=df['quality'], ax=ax1, color='blue',edgecolor = 'b', s=100, marker='o');
ax1.set_title('alcohol', fontsize=13)

sns.scatterplot(x=df['sulphates'], y=df['quality'], ax=ax2, color='green',edgecolor = 'b', s=150, marker='o');
ax2.set_title('sulphates', fontsize=13)

sns.scatterplot(x=df['pH'], y=df['quality'], ax=ax3, color='gold',edgecolor = 'b', s =150, marker='o');
ax3.set_title('pH', fontsize=13)

sns.scatterplot(x=df['density'], y=df['quality'], ax=ax4, color='#FB1861',edgecolor = 'b', s=120, marker='o');
ax4.set_title('density', fontsize=13)

sns.scatterplot(x=df['total sulfur dioxide'], y=df['quality'] , ax=ax5, color='purple', edgecolor='b', s=120, marker='o');
ax5.set_title('total sulfur dioxide', fontsize=13)

plt.show()


# In[237]:


f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(28,5))

sns.scatterplot(x=df['fixed acidity'], y=df['quality'], ax=ax1, color='blue', edgecolor = 'r', s=100, marker='o');
ax1.set_title('total sulfur dioxide', fontsize=13)

sns.scatterplot(x=df['volatile acidity'], y=df['quality'], ax=ax2, color='green', edgecolor = 'b', s=150, marker='o');
ax2.set_title('volatile acidity', fontsize=13)

sns.scatterplot(x=df['citric acid'], y=df['quality'], ax=ax3, color='orange', edgecolor = 'b', s=150, marker='o');
ax3.set_title('citric acid', fontsize=13)

sns.scatterplot(x=df['residual sugar'], y=df['quality'], ax=ax4, color='#FB1861', edgecolor = 'b', s=120, marker='o');
ax4.set_title('residual sugar', fontsize=13)

sns.scatterplot(x=df['chlorides'], y=df['quality'] , ax=ax5, color='purple', edgecolor = 'b', s=120, marker='o');
ax5.set_title('chlorides', fontsize=13)

plt.show()


# In[238]:


f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(28,5))

sns.scatterplot(x=df['density'], y=df['alcohol'], ax=ax1, color='blue', edgecolor = 'r', s=100, marker='o');
ax1.set_title('total.sulfur.dioxide', fontsize=13)

sns.scatterplot(x=df['free sulfur dioxide'], y=df['total sulfur dioxide'], ax=ax2, color='green', edgecolor = 'b', s=150, marker='o');
ax2.set_title('volatile acidity', fontsize=13)

sns.scatterplot(x=df['residual sugar'], y=df['density'], ax=ax3, color='orange', edgecolor = 'b', s=150, marker='o');
ax3.set_title('citric acid', fontsize=13)

sns.scatterplot(x=df['residual sugar'], y=df['total sulfur dioxide'], ax=ax4, color='#FB1861', edgecolor = 'b', s=120, marker='o');
ax4.set_title('residual sugar', fontsize=13)

sns.scatterplot(x=df['fixed acidity'], y=df['density'] , ax=ax5, color='purple', edgecolor = 'b', s=120, marker='o');
ax5.set_title('chlorides', fontsize=13)

sns.scatterplot(x=df['type'], y=df['sulphates'] , ax=ax6, color='purple', edgecolor = 'b', s=120, marker='o');
ax6.set_title('fixed acidity', fontsize=13)

plt.show()


# In[239]:


# df.to_csv('E:/data/data science/data course/DATASET/winequality.csv')


# In[240]:


# df= pd.read_csv('E:/data/data science/data course/DATASET/winequalityF.csv')


# In[241]:


df.loc[df['total sulfur dioxide']>300 ,'quality' ]


# In[242]:


np.where(df['density']>1.01)


# In[243]:


np.where(df['sulphates']>1.75)


# In[244]:


df.loc[(df['citric acid']>1.5) ,:]


# In[245]:


df.loc[(df['chlorides']>0.5) ,:]


# In[246]:


df.loc[(df['free sulfur dioxide']>100) & (df['quality']<5)]


# In[247]:



df.loc[(df['volatile acidity']>1.5) & (df['quality']<5)]


# In[248]:


df=df.drop([277, 1207, 1630, 3836, 2202, 1779, 1402,
            2302, 661, 696, 702, 740, 1362,2303, 3837, 4106, 
            4196,4050, 4055, 4105,5076,556,1432,2516,2718,634,4196] , axis=0).reset_index(drop=True)


# In[249]:


f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(28,5))

sns.scatterplot(x=df['density'], y=df['alcohol'], ax=ax1, color='blue', edgecolor = 'r', s=100, marker='o');
ax1.set_title('total.sulfur.dioxide', fontsize=13)

sns.scatterplot(x=df['free sulfur dioxide'], y=df['total sulfur dioxide'], ax=ax2, color='green', edgecolor = 'b', s=150, marker='o');
ax2.set_title('volatile acidity', fontsize=13)

sns.scatterplot(x=df['residual sugar'], y=df['density'], ax=ax3, color='orange', edgecolor = 'b', s=150, marker='o');
ax3.set_title('citric acid', fontsize=13)

sns.scatterplot(x=df['residual sugar'], y=df['total sulfur dioxide'], ax=ax4, color='#FB1861', edgecolor = 'b', s=120, marker='o');
ax4.set_title('residual sugar', fontsize=13)

sns.scatterplot(x=df['fixed acidity'], y=df['density'] , ax=ax5, color='purple', edgecolor = 'b', s=120, marker='o');
ax5.set_title('chlorides', fontsize=13)

sns.scatterplot(x=df['type'], y=df['sulphates'] , ax=ax6, color='purple', edgecolor = 'b', s=120, marker='o');
ax6.set_title('fixed acidity', fontsize=13)

plt.show()


# In[194]:


sns.pairplot(data = df, hue='quality level',corner=True)


# In[216]:


sns.pairplot(data = df, hue='type', palette='Set2', corner=True);


# In[215]:


sns.pairplot(data = df, hue='quality', palette='Set2', corner=True);


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[572]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
# %matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, ShuffleSplit, RandomizedSearchCV, train_test_split, StratifiedKFold 
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score
from sklearn.neighbors import KNeighborsClassifier  # knn
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


from datetime import datetime, date, time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, average_precision_score,classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer,KNNImputer
from scipy.stats import norm
from scipy import stats


# In[565]:


df=pd.read_csv('E:/data/data science/data course/DATASET/winequalityN.csv')


# In[218]:


df.shape


# In[566]:


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


# In[567]:


df['type']=df['type'].apply(lambda x : 1 if x == 'white' else 2 )
print('unique type:', df['type'].nunique())


# In[227]:


df.isnull().sum()


# In[568]:


imputer=KNNImputer()
df=pd.DataFrame(imputer.fit_transform(df),columns=df.columns)


# In[569]:


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


# In[553]:


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


# In[570]:


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


# ## Wine Type Classifier Models

# In[250]:


df = df.drop(['total sulfur dioxide'], axis=1)


# In[313]:


scale = StandardScaler()


# In[462]:


X = df.drop(['type', 'quality level'], axis=1)


# In[464]:


y = df['type']


# In[465]:


X = scale.fit_transform(X)


# In[466]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=True)


# In[467]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[468]:


print("mean guality in train: {0:.2f}".format(np.mean(y_train)))
print('mean Quality in test: {0:.2f}'.format(np.mean(y_test)))


# In[469]:


from sklearn.decomposition import PCA


# In[470]:


from sklearn.dummy import DummyClassifier

model_dum = DummyClassifier(strategy='most_frequent')
model_dum.fit(X_train, y_train)
print('score for baseline model: {0:.2f}'.format(model_dum.score(X_test, y_test)))


# In[497]:


algos = {
            'linear_regression': {
                    'model': LogisticRegression(),
                    'params': {
                            'max_iter': [20, 30, 50, 75,100],
                            'multi_class': ['multinomial'],
                            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']
                            }
                    },
            'knn': {
                    'model': KNeighborsClassifier(),
                    'params': {
                            'n_neighbors': [5, 10, 20, 25],
                            'p': [2, 3, 5],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                            }
                    },
            'decision_tree': {
                    'model': DecisionTreeClassifier(),
                    'params': {
                            'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [4, 8, 9, 10],
                            'min_samples_split': [2, 5, 8, 11],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'ccp_alpha': [0, 0.05, 0.1]
                            }
                    },
            'random_forest': {
                    'model': RandomForestClassifier(),
                    'params': {
                            'n_estimators': [100, 300, 500, 600],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [9, 12, 15, 18],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'class_weight': ['balanced', 'balanced_subsample']
                            }
                    },
            }    

   


# In[550]:


scores = []

cv = StratifiedKFold(n_splits=5, random_state=0)
for algo_name, config in algos.items():
    gs = RandomizedSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
    gs.fit(X_train, y_train)
    scores.append({
          'model': algo_name,
          'best_score': gs.best_score_,
          'best_params': gs.best_params_
                  })


# In[499]:


scores


# In[500]:


result = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
result


# In[435]:


lr = KNeighborsClassifier()


# In[436]:


lr.fit(X_train, y_train)


# In[437]:


lr.score(X_test, y_test)


# In[427]:


pca = PCA()
blackbox_model = Pipeline([('pca', pca), ('lr', lr)])
blackbox_model.fit(X_train, y_train)
blackbox_model.score(X_test, y_test)


# ## Wine quality level Classifier Models

# In[542]:


X = df.drop(['quality level'], axis=1)


# In[503]:


y = df['quality level'].apply(lambda x: 1 if x=='Low' else 2 if x=='Medium' else 3)


# In[543]:


X = scale.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=True)


# In[531]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[532]:


print("mean guality in train: {0:.2f}".format(np.mean(y_train)))
print('mean Quality in test: {0:.2f}'.format(np.mean(y_test)))


# In[533]:


from sklearn.dummy import DummyClassifier

model_dum = DummyClassifier(strategy='most_frequent')
model_dum.fit(X_train, y_train)
print('score for baseline model: {0:.2f}'.format(model_dum.score(X_test, y_test)))


# In[544]:


algos = {
            'Logistic_regression': {
                    'model': LogisticRegression(),
                    'params': {
                            'max_iter': [20, 50, 75, 80, 85, 100],
                            'multi_class': ['multinomial'],
                            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                            'l1_ratio': [0, 1]
                            }
                    },
            'knn': {
                    'model': KNeighborsClassifier(),
                    'params': {
                            'n_neighbors': [5, 10, 20, 25],
                            'p': [2, 3, 5],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                            }
                    },
            'decision_tree': {
                    'model': DecisionTreeClassifier(),
                    'params': {
                            'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [4, 8, 9, 10],
                            'min_samples_split': [2, 5, 8, 11],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'ccp_alpha': [0, 0.05, 0.1]
                            }
                    },
            'random_forest': {
                    'model': RandomForestClassifier(),
                    'params': {
                            'n_estimators': [100, 300, 500, 600],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [9, 12, 15, 18],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'class_weight': ['balanced', 'balanced_subsample']
                            }
                    },
            }    

   


# In[547]:


scores = []

cv = StratifiedKFold(n_splits=5, random_state=0)
for algo_name, config in algos.items():
    gs = RandomizedSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
    gs.fit(X_train, y_train)
    scores.append({
          'model': algo_name,
          'best_score': gs.best_score_,
          'best_params': gs.best_params_
                  })
     


# In[548]:


result = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
result


# In[549]:


scores


# ## Wine quality Classifier Models

# In[571]:


X = df.drop(['quality', 'quality level'], axis=1)
y = df['quality']
X = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=True)


# In[586]:


pip uninstall keras


# In[587]:


pip install keras


# In[588]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


# In[573]:


params = {'nf1':[16, 32, 64, 128], 'nf2':[128, 256, 64], 'ks1':[3, 5], 'ks2':[3, 2],
          'n_units':[64, 128], 'lrt':[0.005, 0.01, 0.001], 'epochs':[10, 20, 30], 
          'batch_size':[30, 50], 'drop':[0, 0.5]}

def bl_model(nf1,nf2,ks1,ks2,n_units,lrt,drop):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=nf1, kernel_size=ks1, strides=1, activation='elu', input_shape=(8,8,1), padding='same'))
    model.add(keras.layers.Conv2D(filters=nf2, kernel_size=ks2, strides=1, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(drop))
    model.add(keras.layers.Dense(n_units, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lrt), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return(model)


final_model = KerasClassifier(bl_model)
cnn = RandomizedSearchCV(final_model, param_distributions=params, n_iter=10, scoring='accuracy', cv=5)
cnn.fit(X_train, y_train)


# In[ ]:





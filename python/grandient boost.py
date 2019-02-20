import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
def square(x):
    return (x ** 2)
from sklearn import datasets
# read csv file directly from a URL and save the results
data = pd.read_csv('D:/hk/hk-project/python/train.csv', index_col=0)
data['n_jobs'] = np.where(data.n_jobs==-1,16,data.n_jobs)
data['l1_ratio'] = np.where(data.penalty=='l1',1,data.l1_ratio)
data['l1_ratio'] = np.where(data.penalty=='none',0,data.l1_ratio)
data['l1_ratio'] = np.where(data.penalty=='l2',0,data.l1_ratio)

test = pd.read_csv('D:/hk/hk-project/python/test.csv', index_col=0)
test['n_jobs'] = np.where(test.n_jobs==-1,16,test.n_jobs)
test['l1_ratio'] = np.where(test.penalty=='l1',1,test.l1_ratio)
test['l1_ratio'] = np.where(test.penalty=='none',0,test.l1_ratio)
test['l1_ratio'] = np.where(test.penalty=='l2',0,test.l1_ratio)
# display the first 5 rows
#print(data.head())
import seaborn as sns
data['in_sq'] = data['n_informative'].map(square)
test['in_sq'] = test['n_informative'].map(square)
data['in_fl'] = data['flip_y'].map(square)
test['in_fl'] = test['flip_y'].map(square)
feature_cols = ['l1_ratio', 'alpha', 'in_fl','max_iter','n_jobs', 'in_sq','n_samples'\
                ,'n_features','n_classes','n_clusters_per_class','n_informative','flip_y','random_state']

test_cols = ['l1_ratio', 'alpha', 'in_fl','max_iter','n_jobs', 'in_sq','n_samples'\
                ,'n_features','n_classes','n_clusters_per_class','n_informative','flip_y','random_state']
# use the list to select a subset of the original DataFrame
X = data[feature_cols]
# select a Series from the DataFrame
y = data['time']
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
params = {'n_estimators': 900, 'max_depth': 7, 'min_samples_split': 4,\
          'learning_rate': 0.01, 'loss': 'ls'}
clf = GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
#fore = RandomForestRegressor(n_estimators =50,max_features='log2',random_state =666,min_samples_leaf=2,n_jobs=1)
print(clf.score(X, y))
#GradientBoostingRegressor?
y_pred = clf.predict(X)
print(clf.predict(test[test_cols]))
from sklearn import metrics
# calculate MSE using scikit-learn
print ("MSE:",metrics.mean_squared_error(y,y_pred))
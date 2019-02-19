import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
def square(x):
    return (x ** 2)
# read csv file directly from a URL and save the results
data = pd.read_csv('D:\hk\python\sample_train.csv', index_col=0)
# display the first 5 rows
#print(data.head())
import seaborn as sns
#%matplotlib inline
# visualize the relationship between the features and the response using scatterplots
#g = sns.pairplot(data, x_vars=['l1_ratio', 'alpha', 'max_iter', 'n_samples', 'n_features'], y_vars='time', height=7, aspect=0.8)
#import matplotlib.pyplot as plt
#plt.show()
# create a python list of feature names
data['alpha_square'] = data['alpha'].map(square)
feature_cols = ['l1_ratio', 'alpha', 'max_iter', 'n_samples', 'n_features','alpha_square']
# use the list to select a subset of the original DataFrame
X = data[feature_cols]
print(X.head())
# select a Series from the DataFrame
y = data['time']
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
data2 = pd.read_csv('D:\hk\python\sample_test.csv', index_col=0)
data2['alpha_square'] = data2['alpha'].map(square)
X_test = data2[feature_cols]
#X_test['max_iter'] = X_test.max_iter * X_test.max_iter
y_pred = linreg.predict(X_test)
print(y_pred)
print(linreg.score(X,y))
from sklearn import metrics
# calculate MSE using scikit-learn
train_pred = linreg.predict(X)
print ("MSE:",metrics.mean_squared_error(y, train_pred))

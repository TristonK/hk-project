import numpy as np
import pandas as pd
def square(x):
    return (x ** 2)
# read csv file directly from a URL and save the results
data = pd.read_csv('D:/hk/hk-project/python/train.csv', index_col=0)
data['n_jobs'] = np.where(data.n_jobs==-1, 8, data.n_jobs)
data['l1_ratio'] = np.where(data.penalty=='l2', 0,data.l1_ratio)
data['l1_ratio'] = np.where(data.penalty=='none', 0,data.l1_ratio)
data['l1_ratio'] = np.where(data.penalty=='l1', 1,data.l1_ratio)
#print loc 14
#print(data.loc[14])
import seaborn as sns
# visualize the relationship between the features and the response using scatterplots
g = sns.pairplot(data, x_vars=['n_informative','l1_ratio', 'alpha', 'max_iter', 'n_samples', 'n_features','n_jobs','n_classes','n_clusters_per_class','flip_y','scale'], y_vars='time', height=7, aspect=0.8)
import matplotlib.pyplot as plt
plt.show()
# create a python list of feature names
data['sq_info'] = data['n_informative'].map(square)
feature_cols = ['n_informative','sq_info','l1_ratio', 'alpha',
                'max_iter', 'n_samples', 'n_features','n_jobs','n_classes',
                'n_clusters_per_class','flip_y','random_state']
# use the list to select a subset of the original DataFrame
X = data[feature_cols]
#print(X.head())
# select a Series from the DataFrame
y = data['time']
#split the data
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
#"""""
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
print(linreg.score(X,y))
from sklearn import metrics
# calculate MSE using scikit-learn
#print ("MSE:",metrics.mean_squared_error(y_test,y_pred))
#"""""
#get the answer of the test
data2 = pd.read_csv('D:/hk/hk-project/python/test.csv', index_col=0)
data2['n_jobs'] = np.where(data2.n_jobs==-1, 8, data2.n_jobs)
data2['l1_ratio'] = np.where(data2.penalty=='l2', 0,data2.l1_ratio)
data2['l1_ratio'] = np.where(data2.penalty=='none', 0,data2.l1_ratio)
data2['l1_ratio'] = np.where(data2.penalty=='l1', 1,data2.l1_ratio)
data2['sq_info'] = data2['n_informative'].map(square)
test = data2[feature_cols]
y_pred = linreg.predict(test)
print(y_pred)

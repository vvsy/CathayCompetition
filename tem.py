import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn import preprocessing


buy_data = pd.read_csv('train_buy_info.csv')
buy_data = buy_data.fillna(0)
del buy_data['BUY_YEAR']
del buy_data['AGE']
del buy_data['SEX']
del buy_data['OCCUPATION']
del buy_data['CITY_CODE']
del buy_data['MARRIAGE']
print(buy_data.head(5))

# df.name.unique()

print(buy_data.head(5))

#ttype = buy_data['BUY_TYPE']
buttype = {
    "a": 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6
}
# sex = {
#    "a": 0,
#    "b": 1
#}


buy_data['BUY_TYPE'] = buy_data['BUY_TYPE'].map(buttype)
#buy_data['SEX'] = buy_data['SEX'].map(sex)


print(buy_data.head(5))

x = buy_data.drop('BUY_TYPE', axis=1)
y = buy_data['BUY_TYPE']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)


forest = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=6, n_jobs=2)
forest.fit(X_train, y_train)


test_y_predicted = forest.predict(X_test)
accuracy = metrics.accuracy_score(y_test, test_y_predicted)
print(accuracy)
#from sklearn.tree import DecisionTreeClassifier
#tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
#tree.fit(X_train, y_train)


# print(len(y_test))

#from sklearn.tree import DecisionTreeClassifier
#dtree = DecisionTreeClassifier()
#dtree.fit(X_train, y_train)

# df.count(axis='columns')
# print(buy_data.head())
#cust_data = pd.read_csv('train_cust_info.csv')
# print(cust_data.head())

#train_data = pd.read_csv('train_tpy_info.csv')
# print(train_data.head())

#df = pd.concat([train_data, cust_data, buy_data], axis=1, join='outer')


# print(buy_data.head())
# print(len(buy_data))

#predictors = ["AGE", "SEX", "HEIGHT", "WEIGHT", "OCCUPATION", "CHILD_NUM", "BUY_MONTH", "CITY_CODE", "BUDGET", "MARRIAGE"]
#alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
#kf = cross_validation.KFold(buy_data.shape[0], n_folds=3, random_state=1)
#scores = cross_validation.cross_val_score(alg, buy_data[predictors], buy_data["BUY_TYPE"])
# print(scores.mean())
# a = buy_data['HEIGHT'].mean()

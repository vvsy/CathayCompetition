from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import metrics

# #i mport data # #


# import train_buy_info
dfbuy = pd.read_csv('train_buy_info.csv')
dfbuy = dfbuy.fillna(0)
# import test_buy__x_info
dfbuyt = pd.read_csv('test_buy_x_info.csv')
dfbuyt = dfbuyt.fillna(0)

# 丟掉用不到的變數
df1 = dfbuy.drop(['CUST_ID'], axis=1)
df1t = dfbuyt.drop(['CUST_ID'], axis=1)

##
df2 = df1.drop(['BUY_TYPE'], axis=1)
# 將類別資料數字化
strbuy = df2.select_dtypes(exclude=[np.number])
for i in strbuy:
    df1[i], _ = pd.factorize(df1[i])
###
strbuyt = df1t.select_dtypes(exclude=[np.number])
for i in strbuyt:
    df1t[i], _ = pd.factorize(df1t[i])

#########


###
# 利用df2的變數名稱做特徵變數
df2 = df2.drop(['HEIGHT', 'WEIGHT', 'OCCUPATION', 'CHILD_NUM',
                'BUY_MONTH', 'BUY_YEAR', 'MARRIAGE', 'CITY_CODE'], axis=1)
features = df2.columns
print(features)


##############
# 設定RandomForest
rf = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=10, n_jobs=2)

# fit RandomForest
rf.fit(df1[features], df1['BUY_TYPE'])


# 預測outcome

predd = rf.predict(df1t[features])

outcome = pd.read_csv('Submmit_Sample_testing_Set.csv')
outcome['BUY_TYPE'] = predd

outcome.to_csv("outcome1", sep='\t', index=False)

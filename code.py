from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import metrics

# #i mport data # #

# import train_tpy_info
dftpy = pd.read_csv('train_tpy_info.csv')

# 將類別資料數字化
strdftpy = dftpy.select_dtypes(exclude=[np.number])
for i in strdftpy:
    dftpy[i], _ = pd.factorize(dftpy[i])

# import train_cust_info
dfcust = pd.read_csv('train_cust_info.csv')
# 丟掉太多NA的columns
dfcust = dfcust.drop(['IS_NEWSLETTER', 'CHARGE_WAY', 'INTEREST1', 'INTEREST2', 'INTEREST3', 'INTEREST4', 'INTEREST5', 'INTEREST6', 'INTEREST7', 'INTEREST8', 'INTEREST9', 'INTEREST10', 'STATUS1', 'STATUS2', 'STATUS3', 'STATUS4'], axis=1)
# 將類別資料數字化
strcust = dfcust.select_dtypes(exclude=[np.number])
for i in strcust:
    dfcust[i].fillna(dfcust[i].mode()[0], inplace=True)
    dfcust[i], _ = pd.factorize(dfcust[i])

# import train_buy_info
dfbuy = pd.read_csv('train_buy_info.csv')
dfbuy = dfbuy.fillna(0)
# 合併資料
df1 = pd.concat([dfbuy, dfcust, dftpy], axis=1)
# 丟掉用不到的變數
df1 = df1.drop(['CUST_ID'], axis=1)


# #########變數處理######### #


df1['CHILD_NUM'] = df1['CHILD_NUM'].astype(str)

m = df1['CHILD_NUM'].str.contains("0", regex=False)

df1['child'] = np.select([m], ['a'], default="b")
####

df1['BEHAVIOR'] = (list(zip(df1["BEHAVIOR_1"], df1["BEHAVIOR_2"], df1["BEHAVIOR_3"])))
####

df1['OCCUPATION'] = df1['OCCUPATION'].apply(lambda x: x[:2])
####
df1['HEIGHT'][df1['SEX'] == "a"] = (df1['HEIGHT'] * 50 + 161) * 0.01
df1['HEIGHT'][df1['SEX'] == "b"] = (df1['HEIGHT'] * 50 + 174) * 0.01
df1['WEIGHT'][df1['SEX'] == "a"] = df1['WEIGHT'] * 50 + 58
df1['WEIGHT'][df1['SEX'] == "b"] = df1['WEIGHT'] * 50 + 70

df1['BMI'] = df1['WEIGHT'] / (df1['HEIGHT'] ** 2)

print(df1.head())

# #########變數處理######### #

df2 = df1.drop(['BUY_TYPE'], axis=1)
# 將剩餘類別資料數字化
strbuy = df2.select_dtypes(exclude=[np.number])
for i in strbuy:
    df1[i], _ = pd.factorize(df1[i])

#########
# 區分train,test資料比例
df1['is_train'] = np.random.uniform(0, 1, len(df1)) <= .75
train, test = df1[df1['is_train'] == True], df1[df1['is_train'] == False]

###
# 利用df2的變數名稱做特徵變數
df2 = df2.drop(['CITY_CODE', 'BUY_MONTH', 'HEIGHT', 'WEIGHT', 'BEHAVIOR_1', 'BEHAVIOR_2', 'BEHAVIOR_3', 'IS_EMAIL', 'IS_PHONE', 'IS_APP', 'IS_SPECIALMEMBER', 'PARENTS_DEAD', 'REAL_ESTATE_HAVE', 'IS_MAJOR_INCOME', 'BUY_TPY1_NUM_CLASS', 'BUY_TPY2_NUM_CLASS', 'BUY_TPY3_NUM_CLASS',
                'BUY_TPY4_NUM_CLASS', 'BUY_TPY5_NUM_CLASS', 'BUY_TPY6_NUM_CLASS',
                'BUY_TPY7_NUM_CLASS', 'CHILD_NUM', 'child', 'MARRIAGE', 'BEHAVIOR', 'EDUCATION', 'OCCUPATION', 'BUY_YEAR', 'BMI'], axis=1)
features = df2.columns
print(features)


##############
# 設定RandomForest
rf = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=10, n_jobs=2)

# fit RandomForest
rf.fit(train[features], train['BUY_TYPE'])


# Test if important
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
print(feature_importances)

# 預測outcome
predd = rf.predict(test[features])

# 準確度
accuracy = metrics.accuracy_score(test['BUY_TYPE'], predd)
print(accuracy)
preMetric = pd.crosstab(test['BUY_TYPE'], predd, rownames=['actual'], colnames=['preds'])
print(preMetric)


#
# df1['HEIGHT'][df1['SEX'] == "a"] = (df1['HEIGHT'] + 161) * 0.01
# df1['HEIGHT'][df1['SEX'] == "b"] = (df1['HEIGHT'] + 174) * 0.01
# df1['WEIGHT'][df1['SEX'] == "a"] = df1['WEIGHT'] + 58
# df1['WEIGHT'][df1['SEX'] == "b"] = df1['WEIGHT'] + 70

# df1['BMI'] = df1['WEIGHT'] / (df1['HEIGHT'] ** 2)
#

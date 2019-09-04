import pandas as pd
import numpy as np


df = pd.read_csv('train_buy_info.csv')
df = df.fillna(0)

print(df.HEIGHT[df.SEX == "b"].mean())

df['HEIGHT'][df['SEX'] == "a"] = (df['HEIGHT'] + 161) * 0.01
df['HEIGHT'][df['SEX'] == "b"] = (df['HEIGHT'] + 174) * 0.01
df['WEIGHT'][df['SEX'] == "a"] = df['WEIGHT'] + 58
df['WEIGHT'][df['SEX'] == "b"] = df['WEIGHT'] + 70

df['BMI'] = df['WEIGHT'] / (df['HEIGHT'] ** 2)
print(df)

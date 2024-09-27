import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score




data = pd.read_csv('/Users/rjpathak/Downloads/Housing.csv')
data.head()
print(data.shape)
print(data.info)

print(data.describe())
print(data.isnull().sum())
plt.subplot(2,3,1)
sns.boxplot(x = 'mainroad', y = 'price', data = data)
plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom', y = 'price', data = data)
plt.subplot(2,3,3)
sns.boxplot(x = 'basement', y = 'price', data = data)
plt.subplot(2,3,4)
sns.boxplot(x = 'hotwaterheating', y = 'price', data = data)
plt.subplot(2,3,5)
sns.boxplot(x = 'airconditioning', y = 'price', data = data)
plt.subplot(2,3,6)
sns.boxplot(x = 'furnishingstatus', y = 'price', data = data)
plt.show()

varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
data[varlist] = data[varlist].apply(lambda x: x.map({'yes' : 1, 'no': 0}))
status = pd.get_dummies(data['furnishingstatus'], drop_first=True).astype(int)
data = pd.concat([data, status], axis=1)
data = data.drop('furnishingstatus', axis =1)

df_train, df_test = train_test_split(data, train_size = 0.7, random_state = 100)

scaler = MinMaxScaler()
num_vars = ['area', 'bedrooms', 'price', 'bathrooms', 'stories', 'parking']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
sns.heatmap(df_train.corr(), annot=True, cmap='YlGnBu')
plt.show()

y_train = df_train.pop('price')
X_train = df_train
X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm)
lr_model = lr.fit()
print(lr_model.summary())

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)
X = X_train.drop(['semi-furnished', 'bedrooms'], axis = 1)
X_train_sm = sm.add_constant(X)
lr = sm.OLS(y_train, X_train_sm)
lr_model = lr.fit()
print(lr_model.summary())

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)

y_train_pred = lr_model.predict(X_train_sm)
res = y_train-y_train_pred
sns.displot(res)

df_test[num_vars] = scaler.transform(df_test[num_vars])

y_test = df_test.pop('price')
X_test = df_test

X_test_sm = sm.add_constant(X_test)
X_test_sm = X_test_sm.drop(['semi-furnished', 'bedrooms'], axis = 1)

y_test_pred = lr_model.predict(X_test_sm)

print(r2_score(y_true= y_test, y_pred= y_test_pred))
print(y_test_pred)
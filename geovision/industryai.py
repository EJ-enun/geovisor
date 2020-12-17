#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 03:29:08 2020

@author: cinema
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.datasets import load_boston
from sklearn.ensemble import DecisionTreeClassifier

data = pd.read_excel('/Users/cinema/Desktop/sample_dataset.xlsx')
boston = load_boston()
df = pd.DataFrame(boston.data)
df_price = pd.DataFrame(boston.target)
df.columns = boston.feature_names
print(boston.feature_names)
print(boston.keys())
print(boston.DESCR)
#df.head()

X, y = df.iloc[:,:-1],df.iloc[:,-1]
data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear',colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10,n_estimators = 10)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train, y_train)


from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)


gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)
#errors = [mean_squared_error(y_val, y_pred)for y_pred in gbrt.staged_predict(X_val)]
#bst_n_estimators = np.argmin(errors) + 1
#gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
#gbrt_best.fit(X_train, y_train)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""# Data Preprocessing"""

#%% DATA PREPROCESSING

from data_cleaner import get_cleaned_data
df = get_cleaned_data("final_training_xy_2.csv")


"""# XGBoost Regressor"""
# For RF, X variable columns needs to exclude date and y
x_cols = df.columns.tolist()
x_cols.remove('m_ret_next') if 'm_ret_next' in x_cols else None
x_cols.remove('comp_id')

X = df[x_cols].copy()
y = df.m_ret_next 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


train_dates, test_dates = df.index.unique()[:45], df.index.unique()[45:]
train_data = df.loc[train_dates]
test_data = df.loc[test_dates]

X_train = train_data[x_cols].copy()
X_test = test_data[x_cols].copy()
y_train = train_data.m_ret_next 
y_test = test_data.m_ret_next

X_train

"""## Random Search CV"""

#%% Random search CV to find an overall tree structure
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# 参数寻优
kBest = SelectKBest(score_func=f_regression, k=20)
xgb = XGBRegressor()
model = Pipeline([('feature_selection', kBest), ('xgboost', xgb)])
seed = np.random.seed(42)

# 设置参数调优范围
k_vals = [int(x) for x in np.linspace(30, 40, 3)]
#n_est = [100, 150, 200]
n_est = [150]
max_depth = [3, 4, 5, 6, 7]
min_child_weight = [1, 3, 5]
gamma = [0.2, 0.5, 0.8]
subsample = [0.9]
#learning_rate = [0.05, 0.1]
learning_rate = [0.05]
params = dict(feature_selection__k=k_vals, xgboost__n_estimators=n_est, xgboost__max_depth=max_depth, xgboost__min_child_weight = min_child_weight,
              xgboost__gamma=gamma, xgboost__subsample=subsample, xgboost__learning_rate=learning_rate)
bestModel = RandomizedSearchCV(model, param_distributions=params, cv=10, 
                               scoring='neg_mean_squared_error', n_jobs=-1)

# bestModel = GridSearchCV(model, param_grid=params, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
bestModel.fit(X_train, y_train)
bestParams = bestModel.best_params_

y_pred = bestModel.predict(X_test)
mse = mean_squared_error(y_pred, y_test)
print("Mean Squared Error on the test set is {}".format(mse))

finalModel = model.set_params(**bestParams).fit(X, y)
joblib.dump(finalModel, 'XGBoostReg_return.dat')
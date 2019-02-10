import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

train_df= pd.read_csv('data/processed_train.csv')
test_df = pd.read_csv('data/processed_test.csv')
target_col = "target"

cols_to_use=['feature_1', 'feature_2', 'feature_3','year','month','num_hist_transactions','sum_hist_trans', 'mean_hist_trans',
             'std_hist_trans', 'min_hist_trans', 'max_hist_trans','num_merch_transactions','sum_merch_trans', 'mean_merch_trans',
             'std_merch_trans','min_merch_trans', 'max_merch_trans']

train_X = train_df[cols_to_use]
test_X = test_df[cols_to_use]
train_y = train_df[target_col].values

'''
#tune 1:
depth = [5]
learning = [0.05]
estimators = [100]
gamma = [0.1,0.3,0.5]
min_child_weight =[1,3,5]
#best: Best: -14.493511 using {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5}
#Best: -14.492914 using {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5, 'gamma': 0.3, 'min_child_weight': 5}

#########
model = XGBRegressor(random_state =0)
param_grid = dict(max_depth = depth,learning_rate=learning, n_estimators=estimators, gamma = gamma,min_child_weight=min_child_weight)
kfold = KFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold,verbose=True)
grid_result = grid_search.fit(train_X, train_y)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_,grid_result.best_params_))
print("RMSE: " + math.sqrt(abs(grid_result.best_score_)))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''
model_A= XGBRegressor(max_depth=5,
                      n_estimators=100,
                      learning_rate=0.05,
                      gamma = 0.3,
                      min_child_weight =5,
                      random_state=0
                        )

model_A.fit(train_X, train_y)
a_preds = model_A.predict(test_X)

sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})
sub_df["target"] = a_preds
sub_df.to_csv("results/xgb.csv", index=False)

'''
model.fit(X_train_A, y_train_A)
a_preds = model_A.predict_proba(X_test_A)
a_sub = make_country_sub(a_preds, X_test_A, 'A')
'''
#version 1

# Initialize XGB and GridSearch
'''
xgb = XGBRegressor(nthread=-1) 

grid = GridSearchCV(xgb, params, scoring="neg_mean_squared_error")
grid.fit(train_X, train_y)

print(r2_score(Y_Val, grid.best_estimator_.predict(X_Val))) 
'''

#version 2
'''
def run_xgb(train_X, train_y, val_X, val_y, test_X):
	params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}
	evals_result = {}
	xgb = XGBRegressor(nthread=-1, silent=False)
'''

#version 3
'''
pred_test = 0
kf = model_selection.KFold(n_splits=5, random_state = 2018, shuffle=True)
for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    model.fit(dev_X, dev_y)
    model.predict
    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
    
pred_test/=5
'''
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

train_df = pd.read_csv('data/processed_train.csv')
test_df = pd.read_csv('data/processed_test.csv')
target_col = "target"

cols_to_use=['feature_1', 'feature_2', 'feature_3','year','month','num_hist_transactions','sum_hist_trans', 'mean_hist_trans',
             'std_hist_trans', 'min_hist_trans', 'max_hist_trans','num_merch_transactions','sum_merch_trans', 'mean_merch_trans',
             'std_merch_trans','min_merch_trans', 'max_merch_trans']

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params ={
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "min_child_weight" : 50,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label = train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

train_X = train_df[cols_to_use]
test_X = test_df[cols_to_use]
train_y = train_df[target_col].values

pred_test = 0
kf = model_selection.KFold(n_splits=5, random_state = 2018, shuffle=True)
for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
    
pred_test/=5

sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})
sub_df["target"] = pred_test
sub_df.to_csv("results/baseline_lgb.csv", index=False)
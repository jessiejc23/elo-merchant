import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import gc

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

#stratified_lgb_22112020_0955 received a score of 3.70, previously it was 3.76

data = input("Name of training data: ")
train_df = pd.read_csv('../data/' + data +'.csv')
del data
gc.collect()


data = input("Name of test data: ")
test_df = pd.read_csv('../data/'+ data +'.csv')
target_col = "target"

cols_to_use= [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params ={
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 100,
        "min_child_weight" : 50,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_freq" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label = train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)
    best_score = (model.best_score['valid_0']['rmse']).astype('float64')
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result, best_score

train_X = train_df[cols_to_use]
test_X = test_df[cols_to_use]
train_y = train_df[target_col].values

pred_test = 0
best_score = 0
kf = model_selection.StratifiedKFold(n_splits=5, random_state = 2018, shuffle=True)
for fold_, (dev_index, val_index) in enumerate(kf.split(train_df, train_df['outliers'].values)):
    print("fold {}".format(fold_))
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    pred_test_tmp, model, evals_result,best_score_tmp = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
    best_score += best_score_tmp

    
pred_test/=5
print(best_score/5)

sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})
sub_df["target"] = pred_test
sub_df.to_csv("../results/stratified_lgb_" + pd.datetime.now().strftime("%d%m%Y_%H%M") + '.csv',index=False)
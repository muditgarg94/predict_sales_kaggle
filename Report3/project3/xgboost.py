import warnings
warnings.filterwarnings('ignore')
import gc
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from itertools import product
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.pyplot import figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#Importing datasets
sales_train = pd.read_csv("datamining-project/sales_train.csv")
test = pd.read_csv("datamining-project/test.csv")
test.drop('ID',axis=1,inplace=True)

test_shop_item_pair = test[(test.shop_id==10)]
test_shop_item_pair.loc[test_shop_item_pair.shop_id == 10, 'shop_id']= 11
test.loc[test.shop_id == 10, 'shop_id']= 11



def reduce_size(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

# to get processed data
all_data1 = pd.read_csv('all_data.csv')


feat_to_keep = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month',
       'item_category_id', 'subtype', 'db_avg_items_sold_lag_1',
       'db_shop_cat_avg_items_sold_lag_1', 'db_item_id_items_sold_lag_1',
       'db_item_id_items_sold_lag_2', 'db_item_id_items_sold_lag_3',
       'db_item_id_items_sold_lag_6','city_target_enc',
       'item_id_target_enc', 'month_target_enc','db_shop_city_avg_items_sold_lag_1',
       'db_shop_city_avg_items_sold_lag_2', 'db_city_avg_items_sold_lag_1',
       'month', 'item_months_since_first_sale','item_shop_last_sale',
       'db_item_avg_price_lag_1','delta_price_lag', 'delta_price_lag_1',
        'delta_price_lag_3','max_cnt_lag_1', 'max_cnt_lag_3','max_cnt_lag_6',  'revenue_shop_lag_2']



all_data1 = all_data1.loc[:,feat_to_keep]

X_train = all_data1[all_data1.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = all_data1[all_data1.date_block_num < 33]['item_cnt_month']
X_valid = all_data1[all_data1.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = all_data1[all_data1.date_block_num == 33]['item_cnt_month']
X_test = all_data1[all_data1.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del all_data1
gc.collect();




xgb = xgb.XGBRegressor( max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.9, 
    subsample=0.8, 
    eta=0.1,    
    seed=1)


   

xgb.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        eval_metric = 'rmse', early_stopping_rounds =10,verbose=True)


Y_test = xgb.predict(X_test).clip(0., 20.)
results = X_test.loc[:,['shop_id', 'item_id']]
results['prediction'] = Y_test
if len(test.columns)==3:
	test.drop('ID',axis=1,inplace=True)
sub = pd.merge(test, results, on = ['shop_id', 'item_id'], how='left')
submission = pd.DataFrame({"ID": test.index, "item_cnt_month": sub['prediction']})
file_name = 'submission.csv'
submission.to_csv(file_name, index=False)
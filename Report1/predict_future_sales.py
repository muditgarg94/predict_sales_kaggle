import pandas as pd
import numpy as np
import os
import datetime as dt
import time
import re
cwd = os.getcwd()
path = ['C:\\Users\\mihir\\Downloads\\predict_sales_kaggle\\']

#import stuff

items = pd.read_csv(path[0] +'items.csv')
shops = pd.read_csv(path[0] +'shops.csv')
cats = pd.read_csv(path[0] +'item_categories.csv')
train = pd.read_csv(path[0] +'sales_train.csv')
# set index to ID to avoid droping it later
test  = pd.read_csv(path[0] +'test.csv').set_index('ID')

print(test)

#motivation: https://www.kaggle.com/dlarionov/feature-engineering-xgboost
#https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask


# importing few modules

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)

#efficient looping
from itertools import product
#Encode labels with value between 0 and n_classes-1.
from sklearn.preprocessing import LabelEncoder
#ploting
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

#xgboost
from xgboost import XGBRegressor
from xgboost import plot_importance

#feature importance
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

import sys
#garbage collector interface
import gc
#object serialization
import pickle
sys.version_info

#pre processing
#quick look at the data
print(train.head(5))
print(train[['item_cnt_day','item_price']].quantile([0,0.0001,0.99,0.999,0.99999,0.999999,1.0]))
print(train.info())
print(train.dtypes)
print(train.isnull().sum())
print(train.isna().sum())
print(train.shape)
print(train.min())
print(train.max())
print(train[train.item_cnt_day>500].count())
print(train[train.item_price>25000].count())

#clipping at 500 and 25000
clp_day = train['item_cnt_day'].clip(-22,500)
clp_price = train['item_price'].clip(0.1,25000)

train['item_price'] = clp_price
train['item_cnt_day'] = clp_day

#similarity finders
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


for i in range(0,shops.shape[0]):
    for j in range(i+1,shops.shape[0]):
        tmp = similar(shops.iloc[i,0],shops.iloc[j,0])
        if tmp >0.75:
            print(i,j)
            print(shops.iloc[i,0],shops.iloc[j,0])
            
# 0 57 & 1 58 & 10 11 are duplicates
li_nm = [[0,57],[1,58],[10,11]]
for i in li_nm:
    train.loc[train.shop_id==i[0],'shop_id'] = i[1]
    test.loc[train.shop_id==i[0],'shop_id'] = i[1]

print(shops)
print(cats)
#copying from source code
shops.loc[shops.shop_name=='Сергиев Посад ТЦ "7Я"','shop_name']='СергиевПосад ТЦ "7Я"'

shops['city']= shops["shop_name"].str.split(' ').map(lambda x:x[0])
shops['cty_code'] = LabelEncoder().fit_transform(shops['city'])
print(shops)
shops_n = shops[['shop_id','cty_code']]

print(items)

cats['ind']=cats['item_category_name'].map(lambda x:1 if "-" in x else 0)
cats['tmp'] = cats['item_category_name'].str.split('-')
cats['tmp1']=cats['item_category_name'].str.split('(')
cats['typ1'] = cats['tmp'].map(lambda x:x[0].strip())
cats['subtyp1'] = cats['tmp'].map(lambda x: x[1].strip() if len(x)>1 else x[0].strip())
cats['typ2'] = cats['tmp1'].map(lambda x:x[0].strip() if len(x)>1 else '')
cats['subtyp2'] = cats['tmp1'].map(lambda x: x[1][:-1].strip() if len(x)>1 else '' )
def cat_nt(data):
    if data['ind']==1:
        return data['typ1']
    else:
        return data['typ2']
cats['typ'] = cats.apply(lambda row: cat_nt(row),axis=1)

def cat_nst(data):
    if data['ind']==1:
        return data['subtyp1']
    else:
        return data['subtyp2']
cats['subtyp'] = cats.apply(lambda row: cat_nst(row),axis=1)

cats['type_code'] = LabelEncoder().fit_transform(cats['typ'])
cats['subty_code'] = LabelEncoder().fit_transform(cats['subtyp'])
cats_n = cats[['item_category_id','type_code', 'subty_code']]




#inherited code from source:

# creating each combination for each month 
#using product from itertools
#then using vstack from numpy
#copying datatypes as it is as they are computation efficient
ts = time.time()
matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    s = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], s.shop_id.unique(), s.item_id.unique())), dtype='int16'))
    
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)
time.time() - ts

train['revenue'] = train['item_price'] *  train['item_cnt_day']
#get aggrevated sum of item_cnt_day
ts = time.time()
grp = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
grp.columns = ['item_cnt_month']
grp.reset_index(inplace=True)
time.time() - ts

#merge with matrix on left join 
matrix = pd.merge(matrix, grp, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
#                                .clip(0,20) # NB clip target here
                                .astype(np.float16))

#setting test dataset
col_dtype = ['np.int8','np.int8','np.int16']

test[cols[0]] = 34

for i in range(0,len(cols)):
    test[cols[i]]= test[cols[i]].astype(eval(col_dtype[i]))


matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month

# get shop, cat and item features
matrix = pd.merge(matrix,shops_n,on=['shop_id'],how='left')
matrix = pd.merge(matrix,items,on=['item_id'],how='left')
matrix = pd.merge(matrix,cats_n,on=['item_category_id'],how='left')


matrix['cty_code'] = matrix['cty_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subty_code'] = matrix['subty_code'].astype(np.int8)

#lag features

def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

ts = time.time()
matrix = lag_feature(matrix, [1,2,3,6,12,18,24,30], 'item_cnt_month')
time.time() - ts

def get_fea(df,lags,col,sp_col='item_cnt_month'):
    grp_fea=df.groupby(col).agg({sp_col:['mean']})
    nam_tmp = ''
    for i in range(0,len(col)):
        nam_tmp += col[i][0:6]
    grp_fea.columns = [nam_tmp+"ave_cnt"]
    grp_fea.reset_index(inplace=True)
    df = pd.merge(df,grp_fea,on=col,how='left')
    df[nam_tmp+"ave_cnt"] = df [nam_tmp+"ave_cnt"].astype(np.float16)
    df=lag_feature(df,lags,nam_tmp+"ave_cnt")
    df.drop([nam_tmp+"ave_cnt"], axis=1, inplace=True)
    return df

ts = time.time()
matrix = get_fea(matrix, [1], ['date_block_num'])
time.time() - ts

ts = time.time()
matrix = get_fea(matrix, [1,2,3,6,12], ['date_block_num','item_id'])
time.time() - ts

matrix.head(5)

ts = time.time()
matrix = get_fea(matrix, [1,12], ['date_block_num','item_category_id'])
time.time() - ts

ts = time.time()
matrix = get_fea(matrix, [1,12], ['date_block_num', 'shop_id', 'item_category_id'])
time.time() - ts

ts = time.time()
matrix = get_fea(matrix, [1,12], ['date_block_num', 'shop_id', 'type_code'])
time.time() - ts

ts = time.time()
matrix = get_fea(matrix, [1,12], ['date_block_num', 'shop_id', 'subty_code'])
time.time() - ts

ts = time.time()
matrix = get_fea(matrix, [1,12], ['date_block_num', 'cty_code'])
time.time() - ts
ts = time.time()
matrix = get_fea(matrix, [1], ['date_block_num', 'item_id', 'cty_code'])
time.time() - ts

ts = time.time()
matrix = get_fea(matrix, [1], ['date_block_num', 'type_code'])
time.time() - ts

ts = time.time()
matrix = get_fea(matrix, [1], ['date_block_num', 'subty_code'])
time.time() - ts

#adding month and number of days in the month
matrix['month'] = matrix['date_block_num'] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)
matrix['month']=matrix['month'].astype(np.int8)


#item id level mean price
ts = time.time()
itm_prc = train.groupby(['item_id']).agg({'item_price': ['mean']})
itm_prc.columns = ['item_avg_item_price']
itm_prc.reset_index(inplace=True)

matrix = pd.merge(matrix, itm_prc, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

#month level item id level sales
itm_prc_1 = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
itm_prc_1.columns = ['date_item_avg_item_price']
itm_prc_1.reset_index(inplace=True)

matrix = pd.merge(matrix, itm_prc_1, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

time.time() - ts

#for last 6 months calculate lagging price
lags = [1,2,3,4,5,6]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

#get delta standarized delta
for i in lags:
    matrix['delta_price_lag_'+str(i)] = \
        (matrix['date_item_avg_item_price_lag_'+str(i)] - \
                matrix['item_avg_item_price']) / matrix['item_avg_item_price']

ts = time.time()

#get first non-zero value of delta
def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0

#get first trend month lag
def get_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return i
    return 0

matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag_mnt'] = matrix.apply(get_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag_mnt'] = matrix['delta_price_lag_mnt'].astype(np.int8)
matrix['delta_price_lag'].fillna(0, inplace=True)
matrix['delta_price_lag_mnt'].fillna(0, inplace=True)



fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    fetures_to_drop += ['delta_price_lag_'+str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)

time.time() - ts

ts = time.time()
group = train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

matrix = lag_feature(matrix, [1], 'delta_revenue')

matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
time.time() - ts


ts = time.time()
cache = {}
#recency has default value = -1
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
#gets index and row of the dataframe
#AS MATRIX IS SORTED ON ITEM_ID AND SHOP_ID, TAKING ADVANTAGE FOR DETERMINING RECENCY
#https://www.geeksforgeeks.org/python-pandas-dataframe-at/
#https://www.tutorialspoint.com/python_pandas/python_pandas_iteration.htm
for index, row in matrix.iterrows(): 
#item_id and shop_id key
    key = str(row.item_id)+' '+str(row.shop_id)
#first instance key is updated only if not null value is there
#check if key is in cache? If not, then
    if key not in cache:
#if item count month is not zero then update month
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[index, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num         
time.time() - ts    

cache= {}
matrix['item_last_sale'] = -1
matrix['item_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)

for index, row in matrix.iterrows(): 
#item_id and shop_id key
    key = row.item_id
#first instance key is updated only if not null value is there
#check if key is in cache? If not, then
    if key not in cache:
#if item count month is not zero then update month
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[index, 'item_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num         
time.time() - ts    

ts = time.time()
#transform creates first ever entry column
matrix['item_shop_first_sale'] = \
matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = \
matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

#setting value all negative values as -1 
matrix['item_shop_first_sale']= matrix['item_shop_first_sale'].map(lambda x:-1 if x<0 else x)
matrix['item_first_sale']= matrix['item_first_sale'].map(lambda x:-1 if x<0 else x)
matrix['item_first_sale']= matrix['item_first_sale'].astype(np.int8)
matrix['item_shop_first_sale']= matrix['item_shop_first_sale'].astype(np.int8)
time.time() - ts

#getting values which are older than year 
#needs further pondering
ts = time.time()
matrix = matrix[matrix.date_block_num > 11]
time.time() - ts

ts = time.time()
def fill_na(df):
    for col in df.columns:
        #fill all lag values to zero
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)
time.time() - ts

matrix.columns
matrix.info()

ts = time.time()


matrix = matrix\
[[\
'date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'cty_code',\
        'item_category_id', 'type_code', 'subty_code',\
       'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3',\
       'item_cnt_month_lag_6', 'item_cnt_month_lag_12','date_bave_cnt_lag_1',\
       'date_bitem_iave_cnt_lag_1', 'date_bitem_iave_cnt_lag_2',\
       'date_bitem_iave_cnt_lag_3', 'date_bitem_iave_cnt_lag_6',\
       'date_bitem_iave_cnt_lag_12', 'date_bitem_cave_cnt_lag_1',\
       'date_bitem_cave_cnt_lag_12', 'date_bshop_iitem_cave_cnt_lag_1',\
       'date_bshop_iitem_cave_cnt_lag_12', 'date_bshop_itype_cave_cnt_lag_1',\
       'date_bshop_itype_cave_cnt_lag_12', 'date_bshop_isubty_ave_cnt_lag_1',\
       'date_bshop_isubty_ave_cnt_lag_12', 'date_bcty_coave_cnt_lag_1',\
       'date_bcty_coave_cnt_lag_12',\
       'date_bitem_icty_coave_cnt_lag_1', 'date_btype_cave_cnt_lag_1',\
       'date_bsubty_ave_cnt_lag_1', 'month', 'days', 'delta_price_lag',\
       'delta_price_lag_mnt', 'delta_revenue_lag_1', 'item_shop_last_sale',\
       'item_last_sale', 'item_shop_first_sale', 'item_first_sale']\
]

time.time() - ts


#copying memory efficient code stuff
matrix.to_pickle('data.pkl')
del matrix
del cache
del group
del items
del shops
del cats
del train
# leave test for submission
gc.collect();
data = pd.read_pickle('data.pkl')

#using strategy in the code 
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del data
gc.collect();

ts = time.time()

model = XGBRegressor(
    max_depth=8,
    n_estimators=1500,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=34568)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

time.time() - ts

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

Y_pred1 = model.predict(X_valid)
Y_test1 = model.predict(X_test)


submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv(path[0] +'submissions\\xgb_submission34568.csv', index=False)

submission1 = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test1
})

submission1.to_csv(path[0] +'submissions\\xgb_submission34568.csv', index=False)
# save predictions for an ensemble
pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))
pickle.dump(Y_pred1, open('xgb_train1.pickle', 'wb'))
pickle.dump(Y_test1, open('xgb_test1.pickle', 'wb'))


plot_features(model, (10,14))

plot_features(model, (10,14))
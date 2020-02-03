import time
import pandas as pd
import numpy as np
from tqdm import tqdm


sales_train=pd.read_csv('sales_train.csv')
test=pd.read_csv('test.csv')


# MOtivation:
#link1: https://www.kaggle.com/anqitu/feature-engineer-and-model-ensemble-top-10/output
#link2: https://www.kaggle.com/dimitreoliveira/model-stacking-feature-engineering-and-eda


#retereving the data and sorting it

sales_train[sales_train['item_id']==11373][['item_price']].sort_values(['item_price'])
sales_train[sales_train['item_id']==11365].sort_values(['item_price'])

#Correcting the sales value
#putting median of the data set whose value not present

sales_train['item_price'][2909818]=np.nan

sales_train['item_cnt_day'][2909818]=np.nan
sales_train['item_price'][2909818]=sales_train[(sales_train['shop_id'] ==12) & (sales_train['item_id'] == 11373) & (sales_train['date_block_num'] == 33)]['item_price'].median()
sales_train['item_cnt_day'][2909818]=round(sales_train[(sales_train['shop_id'] ==12) & (sales_train['item_id'] == 11373) & (sales_train['date_block_num'] == 33)]['item_cnt_day'].median())
sales_train['item_price'][885138]=np.nan
sales_train['item_price'][885138]=sales_train[(sales_train['item_id'] == 11365) & (sales_train['shop_id'] ==12) & (sales_train['date_block_num'] == 8)]['item_price'].median()


test_nrow=test.shape[0]
sales_train=sales_train.merge(test[['shop_id']].drop_duplicates(), how='inner')
sales_train['date']=pd.to_datetime(sales_train['date'], format='%d.%m.%Y')


# Aggregating data


from itertools import product

# We process the data of all items/shops for each month
grid=[]
for block_num in sales_train['date_block_num'].unique():
	current_shops=sales_train[sales_train['date_block_num']==block_num]['shop_id'].unique()
	current_items=sales_train[sales_train['date_block_num']==block_num]['item_id'].unique()
	grid.append(np.array(list(product(*[current_shops, current_items, [block_num]])),dtype='int32'))

cols=['shop_id', 'item_id', 'date_block_num']
grid=pd.DataFrame(np.vstack(grid), columns=cols,dtype=np.int32)

cols=['shop_id', 'item_id', 'date_block_num']
sales_train['item_cnt_day']=sales_train['item_cnt_day'].clip(0,20)
gb_cnt=sales_train.groupby(cols)['item_cnt_day'].agg(['sum']).reset_iex().rename(columns={'sum': 'item_cnt_month'})
gb_cnt['item_cnt_month']=gb_cnt['item_cnt_month'].clip(0,20).astype(np.int)


train=pd.merge(grid,gb_cnt,how='left',on=cols).fillna(0)
train['item_cnt_month']=train['item_cnt_month'].astype(int)
train=downcast_dtypes(train)
train.sort_values(['date_block_num','shop_id','item_id'],inplace=True)

item=pd.read_csv('items.csv')
train=train.merge(item[['item_id', 'item_category_id']], on=['item_id'], how='left')
test=test.merge(item[['item_id', 'item_category_id']], on=['item_id'], how='left')

item_category=pd.read_csv('item_categories.csv')
list_category=list(item_category.item_category_name)
#processing data for mean encoding
for i in range(0,1):
	list_category[i]='PC Headsets / Headphones'

for i in range(1,8):
	list_category[i]='Access'
	list_category[8]='Tickets (figure)'
	list_category[9]='Delivery of goods'

for i in range(10,18):
	list_category[i]='Consoles'

for i in range(18,25):
	list_category[i]='Consoles Games'
	list_category[25]='Accessories for games'

for i in range(26,28):
	list_category[i]='phone games'

for i in range(28,32):
	list_category[i]='CD games'

for i in range(32,37):
	list_category[i]='Card'

for i in range(37,43):
	list_category[i]='Movie'

for i in range(43,55):
	list_category[i]='Books'

for i in range(55,61):
	list_category[i]='Music'

for i in range(61,73):
	list_category[i]='Gifts'

for i in range(73,79):
	list_category[i]='Soft'

for i in range(79,81):
	list_category[i]='Office'

for i in range(81,83):
	list_category[i]='Clean'
	list_category[83]='Elements of a food'

from sklearn import preprocessing

encoder=preprocessing.LabelEncoder()

item_category['item_cat_id_fix']=encoder.fit_transform(list_category)

train=train.merge(item_category[['item_cat_id_fix', 'item_category_id']], on=['item_category_id'], how='left')
test=test.merge(item_category[['item_cat_id_fix', 'item_category_id']], on=['item_category_id'], how='left')

del item, item_category, grid, gb_cnt

#adding mean encodings of item/shop
# # For Trainset
goal='item_cnt_month'
global_mean= train[goal].mean()
y_tr=train[goal].values
mean_encoded_col=['shop_id','item_id', 'item_category_id', 'item_cat_id_fix']
from sklearn.model_selection import KFold

for col in tqdm(mean_encoded_col):
	trained_col=train[[col] + [goal]]
	corrcoefs=pd.DataFrame(columns=['Cor'])

	#Mean encodings using Kfold
	kf=KFold(n_splits=5, shuffle=False, random_state=0)
	trained_col[col + '_cnt_month_mean_Kfold']=global_mean
	for tr_i, val_i in kf.split(trained_col):
		X_tr, X_val=trained_col.iloc[tr_i], trained_col.iloc[val_i]
		means=X_val[col].map(X_tr.groupby(col)[goal].mean())
		X_val[col + '_cnt_month_mean_Kfold']=means
	trained_col.iloc[val_i]=X_val
	trained_col.fillna(global_mean, inplace=True)
	corrcoefs.loc[col + '_cnt_month_mean_Kfold']=np.corrcoef(y_tr, trained_col[col + '_cnt_month_mean_Kfold'])[0][1]


	item_id_goal_sum=trained_col.groupby(col)[goal].sum()
	item_id_goal_count=trained_col.groupby(col)[goal].count()
	trained_col[col + '_cnt_month_sum']=trained_col[col].map(item_id_goal_sum)
	trained_col[col + '_cnt_month_count']=trained_col[col].map(item_id_goal_count)
	trained_col[col + '_goal_mean_LOO']=(trained_col[col + '_cnt_month_sum'] - trained_col[goal]) / (trained_col[col + '_cnt_month_count'] - 1)
	trained_col.fillna(global_mean, inplace=True)
	corrcoefs.loc[col + '_goal_mean_LOO']=np.corrcoef(y_tr, trained_col[col + '_goal_mean_LOO'])[0][1]


	item_id_goal_mean=trained_col.groupby(col)[goal].mean()
	item_id_goal_count=trained_col.groupby(col)[goal].count()

	trained_col[col + '_cnt_month_mean']=trained_col[col].map(item_id_goal_mean)
	trained_col[col + '_cnt_month_count']=trained_col[col].map(item_id_goal_count)
	alpha=100
	trained_col[col + '_cnt_month_mean_Smooth']=(trained_col[col + '_cnt_month_mean'] *  trained_col[col + '_cnt_month_count'] + global_mean * alpha) / (alpha + trained_col[col + '_cnt_month_count'])
	trained_col[col + '_cnt_month_mean_Smooth'].fillna(global_mean, inplace=True)
	corrcoefs.loc[col + '_cnt_month_mean_Smooth']=np.corrcoef(y_tr, trained_col[col + '_cnt_month_mean_Smooth'])[0][1]

	cumsum=trained_col.groupby(col)[goal].cumsum() - trained_col[goal]
	sumcnt=trained_col.groupby(col).cumcount()
	trained_col[col + '_cnt_month_mean_Expanding']=cumsum / sumcnt
	trained_col[col + '_cnt_month_mean_Expanding'].fillna(global_mean, inplace=True)
	corrcoefs.loc[col + '_cnt_month_mean_Expanding']=np.corrcoef(y_tr, trained_col[col + '_cnt_month_mean_Expanding'])[0][1]
	train=pd.concat([train, trained_col[corrcoefs['Cor'].idxmax()]], axis=1)

	print(corrcoefs.sort_values('Cor'))



#Combining the data
all_data=train
del train, test, trained_col
all_data=downcast_dtypes(all_data)

#Creating lag-based features for item/shop pair
iex_cols=['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix', 'date_block_num']
cols_to_rename=list(all_data.columns.difference(iex_cols))

#print(cols_to_rename)
shift_range=[1, 2, 3, 4, 12]

for month_shift in tqdm(shift_range):
	train_shift=all_data[iex_cols + cols_to_rename].copy()
	train_shift['date_block_num']=train_shift['date_block_num'] + month_shift
	foo=lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
	train_shift=train_shift.rename(columns=foo)
	all_data=pd.merge(all_data, train_shift, on=iex_cols, how='left').fillna(0)

del train_shift

all_data=all_data[all_data['date_block_num'] >= 12] # Don't use old data from year 2013
lag_cols=[col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]]
all_data=downcast_dtypes(all_data)



#Creating features of date
dates_train=sales_train[['date', 'date_block_num']].drop_duplicates()
dates_test=dates_train[dates_train['date_block_num'] == 34-12]
dates_test['date']=dates_test['date'] + pd.DateOffset(years=1)
dates_test['date_block_num']=34
s_all=pd.concat([dates_train, dates_test])
dates_all['dow']=dates_all['date'].dt.dayofweek
dates_all['year']=dates_all['date'].dt.year
dates_all['month']=dates_all['date'].dt.month
dates_all=pd.get_dummies(dates_all, columns=['dow'])
dow_col=['dow_' + str(x) for x in range(7)]
date_features=dates_all.groupby(['year', 'month', 'date_block_num'])[dow_col].agg('sum').reset_iex()
date_features['days_of_month']=date_features[dow_col].sum(axis=1)
date_features['year']=date_features['year'] - 2013

date_features=date_features[['month', 'year', 'days_of_month', 'date_block_num']]
all_data=all_data.merge(date_features, on='date_block_num', how='left')
date_columns=date_features.columns.difference(set(iex_cols))

# Scaling features 
from sklearn.preprocessing import StandardScaler
train=all_data[all_data['date_block_num']!= all_data['date_block_num'].max()]
test=all_data[all_data['date_block_num']== all_data['date_block_num'].max()]
sc=StandardScaler()
to_drop_cols=['date_block_num']
feature_columns=list(set(lag_cols + iex_cols + list(date_columns)).difference(to_drop_cols))
train[feature_columns]=sc.fit_transform(train[feature_columns])
test[feature_columns]=sc.transform(test[feature_columns])
all_data=pd.concat([train, test], axis=0)
all_data=downcast_dtypes(all_data)
del train, test, date_features, sales_train



# First-level model 

#Save `date_block_num`, as we can't use them as features, 
#but will need them to split the dataset into parts
dates=all_data['date_block_num']
last_block=dates.max()
start_first_level_total=time.perf_counter()
scoringMethod='r2'
from sklearn.metrics import mean_squared_error
from math import sqrt



# Train meta-features M=15 (12 + 15=27)
months_to_generate_meta_features=range(27,last_block +1)
mask=dates.isin(months_to_generate_meta_features)
goal='item_cnt_month'
y_all_level2=all_data[goal][mask].values
X_all_level2=np.zeros([y_all_level2.shape[0], 3])


#Now fill `X_train_level2` with metafeatures
slice_start=0

for cur_block_num in tqdm(months_to_generate_meta_features):
	start_cur_month=time.perf_counter()
	cur_X_train=all_data.loc[dates <  cur_block_num][feature_columns]
	cur_X_test= all_data.loc[dates == cur_block_num][feature_columns]
	cur_y_train=all_data.loc[dates <  cur_block_num, goal].values
	cur_y_test= all_data.loc[dates == cur_block_num, goal].values

#Creating arrays of training, testing and goal dataframes to feed into mode
train_x=cur_X_train.values
train_y=cur_y_train.ravel()
test_x=cur_X_test.values
test_y=cur_y_test.ravel()
preds=[]
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from xgboost import XGBRegressor


#xgboost
print 'XGBoost'
# Use only part of features on XGBoost.
xgb_features=['item_cnt','item_cnt_mean', 'item_cnt_std', 'item_cnt_shifted1', 
                'item_cnt_shifted2', 'item_cnt_shifted3', 'shop_mean', 
                'shop_item_mean', 'item_trend', 'mean_item_cnt']
xgb_train=X_train[xgb_features]
xgb_val=X_validation[xgb_features]
xgb_test=X_test[xgb_features]


xgb_model=XGBRegressor(max_depth=8, 
                         n_estimators=500, 
                         min_child_weight=1000,  
                         colsample_bytree=0.7, 
                         subsample=0.7, 
                         eta=0.3, 
                         seed=0)
xgb_model.fit(xgb_train, Y_train)


xgb_train_pred=xgb_model.predict(xgb_train)
xgb_val_pred=xgb_model.predict(xgb_val)
xgb_test_pred=xgb_model.predict(xgb_test)


print('Train rmse:', np.sqrt(mean_squared_error(Y_train, xgb_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_validation, xgb_val_pred)))



#random forest
print 'Random Forest'
# Use only part of features on random forest.
rf_features=['shop_id', 'item_id', 'item_cnt', 'transactions', 'year',
               'item_cnt_mean', 'item_cnt_std', 'item_cnt_shifted1', 
               'shop_mean', 'item_mean', 'item_trend', 'mean_item_cnt']
rf_train=X_train[rf_features]
rf_val=X_validation[rf_features]
rf_test=X_test[rf_features]

rf_model=RandomForestRegressor(n_estimators=50, max_depth=7, random_state=0, n_jobs=-1)
rf_model.fit(rf_train, Y_train)


rf_train_pred=rf_model.predict(rf_train)
rf_val_pred=rf_model.predict(rf_val)
rf_test_pred=rf_model.predict(rf_test)

print('Train rmse:', np.sqrt(mean_squared_error(Y_train, rf_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_validation, rf_val_pred)))

# Ensembling 
pred_list={}
#Second level using linear regression
lr=LinearRegression()

lr.fit(X_train_level2, y_train_level2)

# # Compute R-squared on the train and test sets.
print('RMSE for %s is %f' %('test_preds_lr_stacking', sqrt(mean_squared_error(y_train_level2, lr.predict(X_train_level2)))))
test_preds_lr_stacking=lr.predict(X_test_level2)
train_preds_lr_stacking=lr.predict(X_train_level2)
print('RMSE for %s is %f' %('train_preds_lr_stacking', sqrt(mean_squared_error(y_train_level2, train_preds_lr_stacking))))

pred_list['test_preds_lr_stacking']=test_preds_lr_stacking




submission=pd.read_csv('sample_submission.csv')


prediction_df=pd.DataFrame(test['ID'], columns=['ID'])
prediction_df['item_cnt_month']=test_preds_lr_stacking.clip(0., 20.)
prediction_df.to_csv('submission.csv', iex=False)
prediction_df.head(10)

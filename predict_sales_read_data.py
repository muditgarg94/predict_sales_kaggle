import numpy as np
import pandas as pd
import datetime as dt

training_data=pd.read_csv('sales_train.csv') #date, dateblocknum, shop_id, item_id, item_price, item_cnt_day
items=pd.read_csv('items.csv') #headers are item_name, item_id, item_category_id
item_categories=pd.read_csv('item_categories.csv') #headers are item_category_name, item_category_id
shops=pd.read_csv('shops.csv') #shop_name, shop _id
test_data=pd.read_csv('test.csv') #ID, shop_id, item_id


'''
We need to preprocess the data to convert to item_cnt_month
First convert date into date format

In otder to get the monthly sale of a product in a shop, group by month, shop_id, item_id
Use the columns -> date, item_price, and item_cnt_day
get the avg of item_price and sum the item_cnt_day
'''

training_data.date=training_data.date.apply(lambda x:dt.datetime.strptime(x,'%d.%m.%Y'))
#print training_data.info()

converted_train_data_by_month=training_data.groupby(['date_block_num','shop_id','item_id'])['date','item_price','item_cnt_day'].agg({'date':['min','max'],'item_price':'mean','item_cnt_day':'sum'})
print converted_train_data_by_month
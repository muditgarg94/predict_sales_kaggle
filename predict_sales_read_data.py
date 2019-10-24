import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


#Dickey-Fuller Test
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput



# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    print len(dataset)
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob


training_data=pd.read_csv('sales_train.csv') #date, dateblocknum, shop_id, item_id, item_price, item_cnt_day
items=pd.read_csv('items.csv') #headers are item_name, item_id, item_category_id
item_categories=pd.read_csv('item_categories.csv') #headers are item_category_name, item_category_id
shops=pd.read_csv('shops.csv') #shop_name, shop _id
test_data=pd.read_csv('test.csv') #ID, shop_id, item_id


'''
We need to preprocess the data to convert to item_cnt_month

In otder to get the monthly sale of a product in a shop, group by month, shop_id, item_id

get the sum of item_cnt_day to get itm_cnt_month, take mean of item price 
'''
training_data['date']=pd.to_datetime(training_data['date'],format='%d.%m.%Y')

#check outliers


converted_train_data_by_month=training_data.groupby(['date_block_num','shop_id','item_id']).agg({'item_price':'mean','item_cnt_day':'sum'})
converted_train_data_by_month.columns=['avg_item_price','item_cnt_month']
converted_train_data_by_month.reset_index(inplace=True)


#print converted_train_data_by_month


#Analyze the data
#clubbing the total sales by every month in a year
total_sales=training_data.groupby(['date_block_num']).agg({'item_cnt_day':'sum'})
'''
#print total_sales
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(total_sales)
plt.show()
#clubbing the total sales by every month 
'''

training_data['clubbed_months']=training_data['date_block_num']%12
total_sales_per_month=training_data.groupby(['clubbed_months']).agg({'item_cnt_day':'sum'})
#print total_sales_per_month

#aggregating monthly clubbed data with item_id and shop_id
monthly_sale_across_years=training_data.groupby(['clubbed_months','shop_id','item_id']).agg({'item_price':'mean','item_cnt_day':'sum'})
monthly_sale_across_years.columns=['avg_item_price','item_cnt_month']
monthly_sale_across_years.reset_index(inplace=True)
#print monthly_sale_across_years


'''
plt.xlabel('Month')
plt.ylabel('Sales every month')
plt.plot(total_sales_per_month)
plt.show()
'''
#Observations:
#There is a seasonality and decreasing trend in sales
#This project is a regression based rather than classification
#Decompose a time series . Test statinarity

#test_stationarity(total_sales.iloc[:,0])


# Now remove the trend 
#diff_total_sales=total_sales.diff()
#test_stationarity(diff_total_sales.iloc[:,0])


diff_total_sales=total_sales.diff(periods=12).fillna(0)
test_stationarity(diff_total_sales.iloc[:,0])




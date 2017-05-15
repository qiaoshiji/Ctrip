# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.preprocessing import OneHotEncoder 
import numpy as np

import xgboost as xgb


product_quantity=pd.read_csv('./data/product_quantity.txt',encoding='utf_8_sig')
product_info=pd.read_csv('./data/product_info.txt',encoding='utf_8_sig')
test_raw=pd.read_csv('./data/prediction_lilei_20170320.txt',encoding='utf_8_sig')

test_raw['product_date']=pd.to_datetime(test_raw.product_month)
test_raw['month']=test_raw.product_date.dt.month
test_raw['year']=test_raw.product_date.dt.year

product_quantity.product_date=pd.to_datetime(product_quantity.product_date)
product_quantity['month']=product_quantity.product_date.dt.month
product_quantity['year']=product_quantity.product_date.dt.year
train=product_quantity.groupby(['product_id','year','month'],as_index=False).ciiquantity.sum()

product_info['cooperatedate'][product_info.cooperatedate<'2000-04-01']='2000-04-01'
product_info['cooperatedate']=pd.to_datetime(product_info['cooperatedate'])
product_info['cooperatetime']=pd.to_datetime('2017-02-01')-product_info['cooperatedate']
product_info['cooperatetime']=product_info['cooperatetime'].dt.days
product_info['voters'][product_info['voters']==-1]=0
product_info['voters_time']=product_info['voters']/product_info['cooperatetime']
product_info['maxstock'][product_info['maxstock']==0]=-1
product_info['voters_maxstock']=product_info['voters']/product_info['maxstock']

h_day_dict={'2014-1':8,'2014-2':11,'2014-3':10,'2014-4':9,'2014-5':10,'2014-6':10,
            '2014-7':8,'2014-8':10,'2014-9':8,'2014-10':12,'2014-11':10,'2014-12':8,
            '2015-1':9,'2015-2':11,'2015-3':8,'2015-4':9,'2015-5':11,'2015-6':9,
            '2015-7':8,'2015-8':10,'2015-9':9,'2015-10':13,'2015-11':9,'2015-12':8,
            '2016-1':11,'2016-2':11,'2016-3':8,'2016-4':10,'2016-5':10,'2016-6':9,
            '2016-7':10,'2016-8':8,'2016-9':8,'2016-10':13,'2016-11':8,'2016-12':9,
            '2017-1':12}
base_time_dict={}
for i in h_day_dict:
    base_time_dict[i]=int(i[5:])+(int(i[:4])-2014)*12

product_sales_info=product_quantity.groupby(['product_id'],as_index=False).mean()
product_sales_info=product_sales_info[['product_id','orderattribute1','orderattribute2','orderattribute3','orderattribute4']]
product_info=pd.merge(product_info,product_sales_info,how='left',on='product_id')
product_info=product_info.fillna(-1)

feature=product_info[['product_id', 'district_id1','district_id2','lat', 'lon', 'eval', 'eval2', 'eval3',
       'eval4','voters','maxstock','orderattribute1','orderattribute2','orderattribute3',
       'orderattribute4','voters_time','voters_maxstock','cooperatetime','cooperatedate']]


for i in ['district_id1','district_id2']:
    feature[i][feature[i]==-1]=0
enc = OneHotEncoder()
enc.fit(np.array(feature[['district_id1','district_id2']]))


one_hot_array = enc.transform(np.array(feature[['district_id1','district_id2']])).toarray()
one_hot_df=pd.DataFrame(one_hot_array,columns=['feature'+str(i) for i in range(len(one_hot_array[0]))])
one_hot_df['product_id']=feature['product_id']
feature=pd.merge(feature,one_hot_df,how='left',on='product_id')

test=test_raw[['product_id','year','month']]
train_test=pd.concat([train,test])
train_test['date']=map(lambda a,b:str(a)+'-'+str(b),train_test['year'],train_test['month'])
train_test['base_time']=train_test['date'].apply(lambda x:base_time_dict[x])
train_test['h_day']=train_test['date'].apply(lambda x:h_day_dict[x])

train_test=pd.merge(train_test,feature,how='left',on='product_id')
train_test['orderattribute1'][train_test['orderattribute1']==-1]=0

enc = OneHotEncoder()
enc.fit(np.array(train_test[['month','orderattribute1']]))


month_one_hot_array = enc.transform(np.array(train_test[['month','orderattribute1']])).toarray()
month_one_hot_df=pd.DataFrame(month_one_hot_array,columns=['month_feature'+str(i) for i in range(len(month_one_hot_array[0]))])
train_test=pd.concat([train_test,month_one_hot_df],axis=1)
train_test.date=pd.to_datetime(train_test.date)
train_test['after_co_time']=train_test.date-train_test.cooperatedate
train_test['after_co_time']=train_test['after_co_time'].dt.days

train=train_test[train_test.ciiquantity.notnull()]
test=train_test[train_test.ciiquantity.isnull()]

feature_label=[i for i in train_test.columns if i not in ['ciiquantity','year','date','district_id1','cooperatedate']]


xgtrain = xgb.DMatrix(train[feature_label], label=train['ciiquantity'])
xgtest = xgb.DMatrix(test[feature_label])

params={'booster':'gbtree',
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'max_depth':8,
	#'lambda':100,
	'subsample':0.9,
	'colsample_bytree':0.9,
	'min_child_weight':30,#8~10
	'eta': 0.1,
	'seed':0,
	#'nthread':12
	}

params['silent'] = 1
watchlist = [ (xgtrain,'train') ]
model = xgb.train(params, xgtrain, 3000, watchlist, early_stopping_rounds=50)

test_y = model.predict(xgtest)

test_raw['ciiquantity_month']=test_y
test_raw['ciiquantity_month'][test_raw['ciiquantity_month']<0]=1
test_raw['ciiquantity_month']=test_raw['ciiquantity_month'].apply(lambda x:int(x))
test_raw=test_raw[['product_id', 'product_month', 'ciiquantity_month']]
test_raw.to_csv('./model_result/model7_result.csv',index=False)


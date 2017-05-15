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
rank_month=train.groupby(['product_id','month'],as_index=False).ciiquantity.mean()
rank_month['ciiquantity_rank']=rank_month.groupby(['product_id']).ciiquantity.rank()
del rank_month['ciiquantity']

product_sales_info=product_quantity.groupby(['product_id'],as_index=False).mean()
product_sales_info=product_sales_info[['product_id','orderattribute1','orderattribute2','orderattribute3','orderattribute4']]

product_price=product_quantity[product_quantity.price>0]
product_price_mean=product_price.groupby(['product_id'],as_index=False).price.mean()
product_price_mean['price_mean_rank']=product_price_mean.price.rank()
product_price_std=product_price.groupby(['product_id'],as_index=False).price.std()
product_price_std.product_id=product_price_mean.product_id
product_price_max=product_price.groupby(['product_id'],as_index=False).price.max()
product_price_min=product_price.groupby(['product_id'],as_index=False).price.min()
product_price_all=pd.merge(product_price_max,product_price_min,how='left',on='product_id')
product_price_all=pd.merge(product_price_all,product_price_mean,how='left',on='product_id')
product_price_all=pd.merge(product_price_all,product_price_std,how='left',on='product_id')
product_price_all.columns=['product_id','price_max','price_min','price_mean','price_mean_rank','price_std']

test=test_raw[['product_id','year','month']]
train_test=pd.concat([train,test])
product_info['upgradedate'][product_info.upgradedate<'1998-01-01']='1997-01-01'
product_info['date']=pd.to_datetime(product_info.upgradedate)
product_info['upgrade_year']=product_info.date.dt.year
product_info['upgrade_month']=product_info.date.dt.month
product_info['upgrade_year'][product_info.upgradedate=='1997-01-01']=-1
product_info['upgrade_month'][product_info.upgradedate=='1997-01-01']=-1

product_info['startdate'][product_info.startdate<'1981-01-01']='1980-01-01'
product_info['date']=pd.to_datetime(product_info.startdate)
product_info['start_year']=product_info.date.dt.year
product_info['start_month']=product_info.date.dt.month
product_info['start_year'][product_info.startdate=='1980-01-01']=-1
product_info['start_month'][product_info.startdate=='1980-01-01']=-1

product_info['cooperatedate'][product_info.cooperatedate=='-1']='2000-04-01'
product_info['date']=pd.to_datetime(product_info.cooperatedate)
product_info['cooperate_year']=product_info.date.dt.year
product_info['cooperate_month']=product_info.date.dt.month
product_info['cooperate_year'][product_info.cooperatedate=='2000-04-01']=-1
product_info['cooperate_month'][product_info.cooperatedate=='2000-04-01']=-1

product_info['voter_per_product']=product_info['voters']/product_info['maxstock']
product_info=pd.merge(product_info,product_sales_info,how='left',on='product_id')
feature=product_info[['product_id', 'district_id1', 'district_id2', 'district_id3',
       'district_id4', 'lat', 'lon', 'railway', 'airport', 'citycenter',
       'railway2', 'airport2', 'citycenter2', 'eval', 'eval2', 'eval3',
       'eval4', 'voters','maxstock','orderattribute1','orderattribute2',
       'orderattribute3','orderattribute4','start_year','start_month',
       'upgrade_year','upgrade_month','cooperate_year','cooperate_month','voter_per_product']]


for i in ['district_id1','district_id2', 'district_id3','district_id4']:
    feature[i][feature[i]==-1]=0
enc = OneHotEncoder()
enc.fit(np.array(feature[['district_id1','district_id2', 'district_id3','district_id4']]))

one_hot_array = enc.transform(np.array(feature[['district_id1','district_id2', 'district_id3','district_id4']])).toarray()
one_hot_df=pd.DataFrame(one_hot_array,columns=['feature'+str(i) for i in range(len(one_hot_array[0]))])
one_hot_df['product_id']=feature['product_id']
feature=feature[['product_id','lat', 'lon', 'railway', 'airport', 'citycenter',
       'railway2', 'airport2', 'citycenter2', 'eval', 'eval2', 'eval3',
       'eval4', 'voters','maxstock','voter_per_product']]
feature=pd.merge(feature,one_hot_df,how='left',on='product_id')

train_test=pd.merge(train_test,feature,how='left',on='product_id')

train=train_test[train_test.ciiquantity.notnull()]
test=train_test[train_test.ciiquantity.isnull()]

feature_label=[i for i in train.columns if i not in ['ciiquantity','railway','eval3', 'airport', 'citycenter', 'railway2', 'airport2', 'citycenter2']]


xgtrain = xgb.DMatrix(train[feature_label], label=train['ciiquantity'])
xgtest = xgb.DMatrix(test[feature_label])

params={'booster':'gbtree',
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'max_depth':10,
	#'lambda':100,
	'subsample':0.7,
	'colsample_bytree':0.7,
	'min_child_weight':10,#8~10
	'eta': 0.05,
	#'seed':77,
	#'nthread':12
	}

params['silent'] = 1
watchlist = [ (xgtrain,'train') ]
model = xgb.train(params, xgtrain, 500, watchlist, early_stopping_rounds=50)

test_y = model.predict(xgtest)
test_raw['ciiquantity_month']=test_y
test_raw['ciiquantity_month'][test_raw['ciiquantity_month']<0]=1
test_raw['ciiquantity_month']=test_raw['ciiquantity_month'].apply(lambda x:int(x))

test_raw=test_raw[['product_id','product_month','ciiquantity_month']]
test_raw.to_csv('./model_result/model9_result.csv',index=False)




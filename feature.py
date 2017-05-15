# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from dateutil.relativedelta import relativedelta

def load_data():
    #读取文件
    prediction_sample=pd.read_table('./data/prediction_lilei_20170320.txt',sep=',',encoding='utf_8_sig')
    product_info=pd.read_table('./data/product_info.txt',sep=',',encoding='utf_8_sig')
    product_quantity=pd.read_table(r'./data/product_quantity.txt',sep=',',encoding='utf_8_sig')
        
    drop_col=['district_id1','district_id2','district_id3','district_id4','railway','airport',\
                  'citycenter','railway2','airport2','citycenter2','upgradedate']
    product_info=product_info.drop(drop_col,axis=1)
        
    product_info.loc[product_info['cooperatedate']=='-1','cooperatedate']='2014-01-01'
    product_info.loc[product_info['cooperatedate']=='1753-01-01','cooperatedate']='2014-01-01'
    product_info['cooperate_date']=product_info['cooperatedate'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
        
    product_info.loc[product_info['startdate']=='-1','startdate']='2014-01-01'
    product_info.loc[product_info['startdate']=='1753-01-01','startdate']='2014-01-01'
    product_info['start_date']=product_info['startdate'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
        
    #缺失值填补
    product_info=product_info.replace('-1',np.nan)
    product_info=product_info.replace(-1,np.nan)
    product_info['maxstock']=product_info['maxstock'].replace(0,np.nan)
    product_info['lat'].fillna(product_info['lat'].mode()[0], inplace=True)
    product_info['lon'].fillna(product_info['lon'].mode()[0], inplace=True)
    product_info['eval'].fillna(product_info['eval'].mean(), inplace=True)
    product_info['eval2'].fillna(product_info['eval2'].mean(), inplace=True)
    product_info['eval3'].fillna(product_info['eval3'].mean(), inplace=True)
    product_info['eval4'].fillna(product_info['eval4'].mean(), inplace=True)
    product_info['voters'].fillna(product_info['voters'].median(), inplace=True)
    product_info['maxstock'].fillna(product_info['maxstock'].median(), inplace=True)
        
    ###计算评分和##############
    product_info['eval_T']=product_info['eval']+product_info['eval2']+product_info['eval3']
        
    ##对产品位置进行聚类，加入位置特征###################
    kmean = KMeans(n_clusters=9)
    feature=product_info.loc[:,['lat','lon']]
    kmean.fit(feature)
    product_info['location']=kmean.predict(feature)
    #画图可发现位置有明显的前后关系，因此取rank
    center_label=pd.DataFrame(kmean.cluster_centers_)
    center_label.rename(columns={0:'lon',1:'lat'},inplace=True)
    center_label['location']=center_label.index
    center_label['location_rank']=center_label['lat'].rank()-1
    center_label=center_label.drop(['lon','lat'],axis=1)
        
    product_info=pd.merge(product_info,center_label,on='location',how='left')
    product_info=product_info.drop('location',axis=1)
        
    ####修改product_quantity日期
    product_quantity=product_quantity.sort_values(by=['product_id','product_date'])
    product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: datetime.strptime(x[:7],'%Y-%m'))
        
    ###提取产品订单属性1，并加入到产品信息中###
    orderattribute1=product_quantity.loc[:,['product_id','orderattribute1']].drop_duplicates()
    product_info=pd.merge(product_info,orderattribute1,on='product_id',how='left')
    product_info['orderattribute1'].fillna(method='ffill',inplace=True)
        
    ###计算产品月销量,合并产品信息
    product_quantity_M=product_quantity.groupby(['product_id','product_month']).sum()
    product_quantity_M=product_quantity_M.reindex(columns=['ciiquantity']).unstack().fillna(method='backfill',axis=1)
    product_quantity_M=product_quantity_M.fillna(method='ffill',axis=1).stack().reset_index()
    data_quantity=pd.merge(product_quantity_M,product_info,on='product_id',how='left')
        
    ##调整示例表，加入产品信息###
    prediction_sample['product_month']=prediction_sample['product_month'].apply(lambda x :datetime.strptime(x[:7],'%Y-%m'))
    prediction_sample=prediction_sample.sort_values(by=['product_id','product_month'])
    data_quantity_test=pd.merge(prediction_sample,product_info,on='product_id',how='left')
    data_quantity_test.rename(columns={'ciiquantity_month':'ciiquantity'},inplace=True)
        
    ###合并训练与预测训练集
    data_quantity_test.replace('-1',np.nan,inplace=True)
    total_data=pd.concat([data_quantity,data_quantity_test])
        
    ##计算合作时长与运营时长
        
    total_data['cooperate_time']=(total_data['product_month'].apply(lambda x : x+relativedelta(months=1))-total_data['cooperate_date'])\
                                    .dt.days.apply(lambda x :max(x,0))
    total_data['start_time']=(total_data['product_month'].apply(lambda x : x+relativedelta(months=1))-total_data['start_date'])\
                                      .dt.days.apply(lambda x :max(x,0))
    ##计算从运营到合作的时间
    total_data['interval_s_c']=(total_data['cooperate_date']-total_data['start_date']).dt.days.apply(lambda x :max(x,0))
        
    ##计算voter/cooperate_time 和 voter/maxstock
    total_data['T_cooperate_time']=(datetime(2017,2,1)-total_data['cooperate_date']).dt.days.apply(lambda x :max(x,0))
    total_data['T_start_time']=(datetime(2017,2,1)-total_data['start_date']).dt.days.apply(lambda x :max(x,0))
    total_data['voters_time']=total_data['voters']/total_data['T_cooperate_time']
    total_data['maxstock'].fillna(total_data['maxstock'].mode()[0], inplace=True)
    total_data['voters_maxstock']=total_data['voters']/total_data['maxstock']
    total_data=total_data.drop(['cooperate_date','cooperatedate','start_date','startdate'],axis=1)
        
    ###提取节假日特征
    holiday=[]
    holiday_2014=[['2014-01',8],['2014-02',11],['2014-03',10],['2014-04',9],['2014-05',10],['2014-06',10],\
                  ['2014-07',8],['2014-08',10],['2014-09',8],['2014-10',12],['2014-11',10],['2014-12',8]]
    holiday_2015=[['2015-01',9],['2015-02',11],['2015-03',8],['2015-04',9],['2015-05',11],['2015-06',9],\
                  ['2015-07',8],['2015-08',10],['2015-09',9],['2015-10',13],['2015-11',9],['2015-12',8]]
    holiday_2016=[['2016-01',11],['2016-02',11],['2016-03',8],['2016-04',10],['2016-05',10],['2016-06',9],\
                  ['2016-07',10],['2016-08',8],['2016-09',8],['2016-10',13],['2016-11',8],['2016-12',9]]
    holiday_2017=[['2017-01',12]]
    holiday.extend(holiday_2014)
    holiday.extend(holiday_2015)
    holiday.extend(holiday_2016)
    holiday.extend(holiday_2017)
        
    holiday=pd.DataFrame(holiday,columns=['product_month','h_days'])
    holiday['product_month']=holiday['product_month'].apply(lambda x: datetime.strptime(x,'%Y-%m'))
    total_data=pd.merge(total_data,holiday,on='product_month',how='left')
        
    ###计算basetime
    def diffMonth(endDate):  
        startDate=datetime(2013,12,1)
        startYear=startDate.year  
        startMonth=startDate.month  
        endYear=endDate.year  
        endMonth=endDate.month  
        #如果是同年  
        if startYear==endYear:  
            diffmonths=endMonth-startMonth  
        #如果是上年  
        elif endYear-startYear==1:  
            diffmonths=12+endMonth-startMonth  
        #如果是大于1年  
        elif endYear-startYear>1:  
            years=endYear-startYear  
            diffmonths=(years-1)*12+12+endMonth-startMonth  
            #如果开始日期大约结束日期报错  
        elif endYear-startYear<0 or( endYear==startYear and endMonth-startMonth):  
            print('enddate must greater than startdate')
        return int(diffmonths)
    total_data['base_time']=total_data['product_month'].apply(diffMonth)
        
    ##提取月特征
    total_data['month']=total_data['product_month'].apply(lambda x:x.month)
    ##将位置、月份、订单属性1转为哑变量
    location=pd.get_dummies(total_data['location_rank'],prefix='location')
    month=pd.get_dummies(total_data['month'],prefix='month')
    orderattribute1=pd.get_dummies(total_data['orderattribute1'],prefix='orderattribute1')
    new_data=pd.concat([total_data,location,month,orderattribute1],axis=1)
    new_data=new_data.sort_values(by=['product_month','product_id'])
    new_data=new_data.drop(['T_cooperate_time'],axis=1)
        
    X_train=new_data.loc[(new_data['base_time']<24),:].drop(['product_month','ciiquantity'],axis=1)
    y_train=new_data.loc[(new_data['base_time']<24),'ciiquantity']
    X_submission=new_data.loc[new_data['base_time']>=24,:].drop(['product_month','ciiquantity'],axis=1)
    index=new_data.loc[:,['product_id','product_month']]
    
    return  X_train, y_train, X_submission,new_data,index





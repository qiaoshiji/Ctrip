
from __future__ import division

from feature import load_data
import xgboost as xgb
#import lightgbm as lgb

def xgb_model(X,y,X_submission,seed):
    
    dtrain=xgb.DMatrix(X,label=y)
    dtest=xgb.DMatrix(X_submission)
    params={'max_depth':10,
            'eta':0.01,
            'min_child_weight':25,
            'gamma':10,
            'colsample_bytree':0.9,
            'subsample':0.9,
            'seed':seed,
            'objective':'reg:linear'}
    num_round=3000
    model=xgb.train(params,dtrain,num_round)
    y_pred=model.predict(dtest)
    return y_pred


def save_txt(new_data,index,y_pred,address):
    new_data.loc[(new_data['base_time']>=24),'ciiquantity']=y_pred
    pred=new_data.copy()
    ##库存约束#
    logi=(pred['ciiquantity']>pred['maxstock']*30)
    pred.loc[logi,'ciiquantity']=pred.loc[logi,'maxstock']*30
    pred.loc[pred['ciiquantity']<0,'ciiquantity']=0
    #取出预测值##
    pred=pred.loc[(pred['base_time']>=24),['product_id','ciiquantity']]
    ####修整并生成txt##########
    pred['product_month']=index['product_month']
    pred.set_index(pred['product_id'],inplace=True)
    pred=pred.reindex(columns=['product_month','ciiquantity'])
    pred.rename(columns={'ciiquantity':'ciiquantity_month'},inplace=True)
    
    pred['ciiquantity_month']=pred['ciiquantity_month'].apply(round)
    print(pred.head())
    pred.to_csv(address)

if __name__ == '__main__':
    
    X, y, X_submission,new_data,index = load_data()

    y_pred=xgb_model(X,y,X_submission,seed=0)
    address='./model_result/model3_result.csv'
    save_txt(new_data,index,y_pred,address)          


from __future__ import division

from feature import load_data
#import xgboost as xgb
import lightgbm as lgb
    
def lgb_model(X,y,X_submission,seed):

    lgb_train = lgb.Dataset(X, y)
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              'objective': 'regression',
              #'metric': {'l2', 'auc'},
              'num_leaves':64,
              'max_depth':15,
              'learning_rate': 0.01,
              'feature_fraction': 0.9,
              'bagging_fraction': 0.8,
              'bagging_freq': 5,
              'verbose': 0,
              'bagging_seed':seed}
    gbm = lgb.train(params,lgb_train,num_boost_round=3500)
    y_pred = gbm.predict(X_submission)
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

    y_pred=lgb_model(X,y,X_submission,seed=3)
    address='./model_result/model2_result.csv'
    save_txt(new_data,index,y_pred,address)

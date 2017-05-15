
from __future__ import division
import numpy as np
#from sklearn.cross_validation import StratifiedKFold
from feature import load_data
from sklearn.cross_validation import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    X, y, X_submission,new_data,index = load_data()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    #skf = list(StratifiedKFold(y, n_folds))

    skf = list(KFold(len(y), n_folds))
    print(skf)
    for  (train, test) in skf:
        print(train)
        print(test)
        print('*****')
    print(y)
#    clfs = [RandomForestRegressor(max_depth=10,n_estimators=1000),
#            GradientBoostingRegressor(max_depth=10,learning_rate=0.05, subsample=0.9, n_estimators=1000),
    clfs =  [XGBRegressor(seed=2017,max_depth=10,learning_rate=0.01,min_child_weight=25,gamma=10,subsample=0.9,colsample_bytree=0.9,n_estimators=3000),
            LGBMRegressor(seed=2017,max_depth=10, learning_rate=0.01, num_leaves=64 , colsample_bytree=0.9, subsample_freq=5,subsample=0.8, n_estimators=3500)] 

    print("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print("Fold", i)
            X_train = X.ix[train,:]
            y_train = y[train]
            X_test = X.ix[test,:]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict(X_test)
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict(X_submission)
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    new_data.loc[(new_data['base_time']>=24),'ciiquantity']=dataset_blend_test.mean(axis=1)
    
    
    predict=new_data.copy()
    ##库存约束#
    logi=(predict['ciiquantity']>predict['maxstock']*30)
    predict.loc[logi,'ciiquantity']=predict.loc[logi,'maxstock']*30
    predict.loc[predict['ciiquantity']<0,'ciiquantity']=0
    #取出预测值##
    pre=predict.loc[(predict['base_time']>=24),['product_id','ciiquantity']]
    ####修整并生成txt##########
    pre['product_month']=index['product_month']
    pre.set_index(pre['product_id'],inplace=True)
    pre=pre.reindex(columns=['product_month','ciiquantity'])
    pre.rename(columns={'ciiquantity':'ciiquantity_month'},inplace=True)
    
    pre['ciiquantity_month']=pre['ciiquantity_month'].apply(round)
    #print(pre)
#    pre.to_csv(r'F:\数据挖掘学习\携程\prediction_Shusheng1_20170423.txt')
    pre.to_csv(r'./model_result/model1_result.csv')

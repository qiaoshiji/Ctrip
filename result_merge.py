import pandas as pd

model1=pd.read_csv(r'./model_result/model1_result.csv',sep=',')
model2=pd.read_csv(r'./model_result/model2_result.csv',sep=',')
model3=pd.read_csv(r'./model_result/model3_result.csv',sep=',')
model4=pd.read_csv(r'./model_result/model4_result.csv',sep=',')
model5=pd.read_csv(r'./model_result/model5_result.csv',sep=',')
model6=pd.read_csv(r'./model_result/model6_result.csv',sep=',')
model7=pd.read_csv(r'./model_result/model7_result.csv',sep=',')
model8=pd.read_csv(r'./model_result/model8_result.csv',sep=',')
model9=pd.read_csv(r'./model_result/model9_result.csv',sep=',')
model10=pd.read_csv(r'./model_result/model10_result.csv',sep=',')
model11=pd.read_csv(r'./model_result/model11_result.csv',sep=',')

new_result=model1.copy()
new_result['ciiquantity_month']=0.25*model1['ciiquantity_month']+0.25*model2['ciiquantity_month']\
                            +0.22*model3['ciiquantity_month']+0.19*model4['ciiquantity_month']\
                            +0.2*model5['ciiquantity_month']+0.19*((model6['ciiquantity_month']\
                           +model7['ciiquantity_month']+model8['ciiquantity_month']+model9['ciiquantity_month'])/4)\
                            -0.17*model10['ciiquantity_month']-0.13*model11['ciiquantity_month']
                            
new_result.set_index(new_result['product_id'],inplace=True)
new_result=new_result.drop('product_id',axis=1)
new_result.to_csv(r'.\submission_result\submission_result.txt')
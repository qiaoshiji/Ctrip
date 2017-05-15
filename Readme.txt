
1、data存放源数据集，包括product_info.txt，product_quantity.txt，prediction_lilei_20170320.txt。各模型程序均从data里读取数据。

2、feature为最优单模型特征的生成程序，不生成预测结果，model1、model2、model3采用的都是最优单模型特征，因此均调用feature模块。

3、运行各model生成11份预测结果，相应地存放于model_result文件夹中,用于融合。

4、完成3后，运行result_merge程序读取model_result文件夹中的结果数据，并进行加权平均融合，生成最终的提交结果，并存于在submission_result中。

5、不同运行环境，生成的结果可能有差异，提交的预测结果使用的是如下的运行环境：

（1）model1、model6、model7、model8、model9运行环境为python 2.7，xgboost 0.6，lightbgm 0.1

（2）model2、model3、result_merge运行环境为python 3.5.3，xgboost 0.6，lightgbm 0.1

（3）model4、model5、model10、model11运行环境为R语言3.3.2，data.table 1.10.4，xgboost 0.6-4, lubridate 1.5.6, magrittr 1.5, h2o 3.10.3.6

#xgbtree
##无自构建映射
###增加经纬度分类
##增加树深度
##增加订单属性onehot
##增加price统计量
##取前45重要性变量
rm(list=ls())
library(data.table)
library(lubridate)
library(magrittr)
library(xgboost)

#读取数据---------------------------------------------------------------
prediction = fread("./data/prediction_lilei_20170320.txt")
product_info = fread("./data/product_info.txt")
product_quantity= fread("./data/product_quantity.txt")
##各列缺失值个数汇总----------------------------------------------------
for( i in names(product_info)){
  print(paste("na ",i,"=",sum(product_info[,i,with=F]==-1)))
}
for( i in names(product_quantity)){
  print(paste("na ",i,"=",sum(product_quantity[,i,with=F]==-1)))
}

#1.prediction表------------------------------------------------------
prediction$product_month <- lubridate::ymd(prediction$product_month )
prediction$year <- lubridate::year(prediction$product_month )
prediction$month <- lubridate::month(prediction$product_month )
prediction$day <- lubridate::day(prediction$product_month )
setkey(prediction,product_month)
prediction$ciiquantity_month <- NULL
#2.product_info表-----------------------------------------------------------
##处理时间数据
product_info$startdate <- lubridate::ymd(product_info$startdate)#因为有缺失，所以报警告
#计算均值，来填充时间---------------------------------------
inded_na_startdate <- which(is.na(product_info$startdate))
date_mean <- mean(ymd(product_info$startdate[-inded_na_startdate]))
product_info$startdate[inded_na_startdate] <- date_mean #均值填充日期
#startdate填充完毕-------------------------------------------------------------------
product_info$startdate_year <- lubridate::year(product_info$startdate)
product_info$startdate_month <- lubridate::month(product_info$startdate)
product_info$startdate_day <- lubridate::day(product_info$startdate)
product_info$upgradedate <- lubridate::ymd(product_info$upgradedate)
#计算均值，来填充时间----------------------------------------------
inded_na_upgradedate <- which(is.na(product_info$upgradedate))
date_mean <- mean(ymd(product_info$upgradedate[-inded_na_upgradedate]))
product_info$upgradedate[inded_na_upgradedate] <- date_mean #均值填充日期
#upgradedate填充完毕------------------------------------------------------------------
product_info$upgradedate_year <- lubridate::year(product_info$upgradedate)
product_info$upgradedate_month <- lubridate::month(product_info$upgradedate)
product_info$upgradedate_day <- lubridate::day(product_info$upgradedate)
product_info$cooperatedate <- lubridate::ymd(product_info$cooperatedate)
#计算均值，来填充时间----------------------------------------
inded_na_cooperatedate <- which(is.na(product_info$cooperatedate))
date_mean <- mean(ymd(product_info$cooperatedate[-inded_na_cooperatedate]))
product_info$cooperatedate[inded_na_cooperatedate] <- date_mean #均值填充日期
#cooperatedate填充完毕----------------------------------------------
product_info$cooperatedate_year <- lubridate::year(product_info$cooperatedate)
product_info$cooperatedate_month <- lubridate::month(product_info$cooperatedate)
product_info$cooperatedate_day <- lubridate::day(product_info$cooperatedate)

##处理特征
product_info$railway2 <- NULL  #多重共线性,同时缺失过多，4000个中3878个缺失，留一个即可
product_info$citycenter2 <- NULL  #多重共线性,同时缺失过多，4000个中3878个缺失
index_lon_lat_na <- which(product_info[,"lat",with=F]==-1) #104个经纬度缺失
product_info$goods <- ifelse(1:4000 %in% index_lon_lat_na,1,2)
#由于用线性回归lm的R^2发现了voters这个变量相当重要，下面对这个变量进行处理
#先对voters补缺失
product_info$voters <- ifelse(product_info$voters == -1,
                              median(product_info$voters),
                              product_info$voters)
product_info$voters_by_eval3 <- product_info$voters*product_info$eval3#点评人数*用户评分
product_info$voters_by_eval4 <- product_info$voters*product_info$eval4#点评人数*综合评分
#计算各评级差值,在这之前先补缺失,这几个特征还是有用的
#补缺失
product_info$eval2 <- ifelse(product_info$eval2==-1,
                             median(product_info$eval2),
                             product_info$eval2)
product_info$eval3 <- ifelse(product_info$eval3==-1,
                             median(product_info$eval3),
                             product_info$eval3)
product_info$eval4 <- ifelse(product_info$eval4==-1,
                             median(product_info$eval4),
                             product_info$eval4)
product_info$eval_eval2 <- product_info$eval-product_info$eval2
product_info$eval_eval3 <- product_info$eval-product_info$eval3
product_info$eval_eval4 <- product_info$eval-product_info$eval4
product_info$eval2_eval3 <- product_info$eval2-product_info$eval3
product_info$eval2_eval4 <- product_info$eval2-product_info$eval4
product_info$eval3_eval4 <- product_info$eval3-product_info$eval4
#处理经纬度数据进行分类
mydata <- product_info[,.(product_id,lat,lon)]
setkey(mydata,lon)
mydata
mydata$lable <- 10 
mydata$lable[1:495] <- -1 #缺失
mydata$lable[496:1010] <- 1 # 喀麦隆
mydata$lable[1011:1655] <- 2　#中非 
mydata$lable[1656:2050] <- 3  #苏丹
mydata$lable[2051:2880] <- 4  #阿拉伯海
mydata$lable[2881:3580] <- 5  #印度
mydata$lable[3581:3710] <- 7  #中国
mydata$lable[3711:3755] <- 6  #缅甸
mydata$lable[3756:3771] <- 7  #中国
mydata$lable[3772:3799] <- 6  #缅甸
mydata$lable[3800:4000] <- 7  #中国
lat_lon_lable <- mydata[,.(product_id ,lable)]
setkey(lat_lon_lable,product_id)
product_info$lat_lon_lable <- lat_lon_lable $lable
a <- product_quantity[,.SD[c(1)], by=product_id] %>%dplyr::arrange(product_id)%>%as.data.table()
b <- a[,c("product_id","orderattribute1")]
product_info2 <- merge(product_info,b,by="product_id",all.x = T)
product_info2$orderattribute1[is.na(product_info2$orderattribute1)] <- 1
d <- product_quantity[,.(orderattribute2_ratio1=sum(orderattribute2==1)/.N,
                         orderattribute2_ratio2=sum(orderattribute2==2)/.N,
                         orderattribute3_ratio1=sum(orderattribute3==1)/.N,
                         orderattribute3_ratio2=sum(orderattribute3==2)/.N
),by=.(product_id)]
product_info3 <- merge(product_info2,d,by="product_id",all.x = T)
e <- product_quantity[price>-1,.(id_mean_price=mean(price,na.rm = T),
                                 id_max_price=max(price,na.rm = T),
                                 id_min_price=min(price,na.rm = T),
                                 id_sd_price=sd(price,na.rm = T)),by=product_id]
product_info4 <- merge(product_info3,e,by="product_id",all.x = T)
product_info4 <- imputeMissings::impute(product_info4,method = "median/mode")%>%as.data.table()
product_info <- dummies::dummy.data.frame(product_info4,names=c("lat_lon_lable","orderattribute1"), sep="_")%>%as.data.table()

product_info$startdate_before201512 <- ifelse(product_info$startdate>=ymd("2015-12-01"),1,0)
product_info$upgradedate_before201512 <- ifelse(product_info$upgradedate>=ymd("2015-12-01"),1,0)
product_info$cooperatedate_before201512 <- ifelse(product_info$cooperatedate>=ymd("2015-12-01"),1,0)

#dim(product_info )

#3.product_quantity表-----------------------------------------------------------
##处理时间数据
product_quantity$product_date <- ymd(product_quantity$product_date)
product_quantity$year <- lubridate::year(product_quantity$product_date) 
product_quantity$month <- lubridate::month(product_quantity$product_date) 
product_quantity$day <- lubridate::day(product_quantity$product_date)
##处理特征
product_quantity$ordquantity <- NULL #因为不知道预测时的订单量
#排序只是为了方便观察
setcolorder(product_quantity,c("product_id","product_date","ciiquantity",
                               "year", "month","day",
                               "price",
                               names(product_quantity)[3:6]
))
setkey(product_quantity,product_id,year,month,day)

product_quantity_pro = product_quantity[price>-1,.(ciiquantity_month = sum(ciiquantity),
                                                   price_avg = mean(price),
                                                   price_max = max(price),
                                                   price_min = min(price),
                                                   price_sd  = sd(price),
                                                   # orderattribute1=round(mean(orderattribute1)),
                                                   orderattribute2=round(mean(orderattribute2)),
                                                   orderattribute3=round(mean(orderattribute3)),
                                                   orderattribute4=round(mean(orderattribute4))
),
by=.(product_id,year,month)]

product_quantity_pro$price_revest <- 1/(product_quantity_pro$price_avg+2)#价格与订单量成反比


product_quantity_pro$product_month <- paste(product_quantity_pro$year,
                                            product_quantity_pro$month,1,sep= "-") %>%lubridate::ymd()
product_quantity_pro$season <- ifelse(product_quantity_pro$month <=3,1,
                                      ifelse(product_quantity_pro $month <=6,2,
                                             ifelse(product_quantity_pro $month<=9,3,4)))
setcolorder(product_quantity_pro,c("product_id","product_month","ciiquantity_month",
                                   "year", "month","season",names(product_quantity_pro)[5:12]))

train_total<- merge(product_quantity_pro,product_info,by="product_id")
setkey(train_total,product_month)#为了跟测试集顺序一样
dim(train_total)
train_total_nomiss <- imputeMissings::impute(train_total,method = "median/mode")%>%as.data.table()
#评分人数乘以价格
train_total_nomiss$price_avg_by_voters <- train_total_nomiss$price_avg * train_total_nomiss$voters
train_total_nomiss$price_max_by_voters <- train_total_nomiss$price_max * train_total_nomiss$voters
train_total_nomiss$price_min_by_voters <- train_total_nomiss$price_min * train_total_nomiss$voters
train_total_nomiss$price_sd_by_voters <- train_total_nomiss$price_sd * train_total_nomiss$voters

dim(train_total_nomiss)
dummy_orderattribute_train <- dummies::dummy.data.frame(train_total_nomiss[,c("orderattribute2", "orderattribute3","orderattribute4")],
                                                        names=c("orderattribute2","orderattribute3","orderattribute4"), sep="_")
train_total_nomiss <- cbind(train_total_nomiss,dummy_orderattribute_train)
dim(train_total_nomiss)

holiday_train <- as.data.table( ymd(c("2014-01-08","2014-02-11","2014-03-10","2014-04-09","2014-05-10",
                                      "2014-06-10","2014-07-08","2014-08-10","2014-09-08","2014-10-12",
                                      "2014-11-10","2014-12-08",
                                      "2015-01-09","2015-02-11","2015-03-08","2015-04-09","2015-05-11",
                                      "2015-06-09","2015-07-08","2015-08-10","2015-09-09","2015-10-13",
                                      "2015-11-09")))
holiday_train1 <- holiday_train[,.(year=year(V1),month=month(V1),holiday_num=day(V1))]
#ggplot2::ggplot(holiday_train1,aes(x=month,y=holiday_num,colour=as.factor(year)))+geom_point()+geom_line()
train_total_nomiss <- merge(train_total_nomiss,holiday_train1,by=c("year","month"),all.x = T)

dim(train_total_nomiss)
train_total_nomiss$product_date <- ymd(paste(train_total_nomiss$year,train_total_nomiss$month,01))

train_total_nomiss$product_date_reduce_startdate <- train_total_nomiss$product_date-train_total_nomiss$startdate
train_total_nomiss$product_date_reduce_upgradedate <- train_total_nomiss$product_date-train_total_nomiss$upgradedate
train_total_nomiss$product_date_reduce_cooperatedate <- train_total_nomiss$product_date-train_total_nomiss$cooperatedate
train_total_nomiss$product_date_reduce_2014 <- train_total_nomiss$product_date-ymd("2014-01-01")
train_total_nomiss$product_date_reduce_startdate <- ifelse(train_total_nomiss$product_date_reduce_startdate<0,0,train_total_nomiss$product_date_reduce_startdate)
train_total_nomiss$product_date_reduce_upgradedate <- ifelse(train_total_nomiss$product_date_reduce_upgradedate<0,0,train_total_nomiss$product_date_reduce_upgradedate)
train_total_nomiss$product_date_reduce_cooperatedate <- ifelse(train_total_nomiss$product_date_reduce_cooperatedate<0,0,train_total_nomiss$product_date_reduce_cooperatedate)

#构造测试集-------------------------------------------------------------
#prediction
#product_quantity_pro
prediction_col3 <- prediction[,.(product_id,product_month,month)]
product_quantity_pro_col8 <- product_quantity_pro[,.(product_id,product_month,price_avg,
                                                     price_max,price_min,price_sd,month,
                                                     orderattribute2,
                                                     orderattribute3,orderattribute4)]
price_add_predict <-product_quantity_pro_col8[,.(price_avg=mean(price_avg,na.rm = T),
                                                 price_max=mean(price_max,na.rm = T),
                                                 price_min=mean(price_min,na.rm = T),
                                                 price_sd=mean(price_sd,na.rm = T),
                                                 #orderattribute1=round(mean(orderattribute1,na.rm = T)),
                                                 orderattribute2=round(mean(orderattribute2,na.rm = T)),
                                                 orderattribute3=round(mean(orderattribute3,na.rm = T)),
                                                 orderattribute4=round(mean(orderattribute4,na.rm = T))
),by=.(product_id ,month)]


test <- merge(prediction_col3 ,price_add_predict,by=c("product_id","month"),all.x = T)
setkey(test,product_month)
test$year <- lubridate::year(test$product_month)

#补缺失
test$price_revest <- 1/(test$price_avg+2)

test$season <- ifelse(test$month <=3,1,
                      ifelse(test$month <=6,2,
                             ifelse(test$month<=9,3,4)))
setcolorder(test,names(product_quantity_pro)[-3])
test_total <- merge(test,product_info,by="product_id",all.x = T)
test_total_nomiss <- imputeMissings::impute(test_total,method = "median/mode")%>%as.data.table()
setkey(test_total_nomiss,product_month)
test_total_nomiss$price_avg_by_voters <- test_total_nomiss$price_avg * test_total_nomiss$voters
test_total_nomiss$price_max_by_voters <- test_total_nomiss$price_max * test_total_nomiss$voters
test_total_nomiss$price_min_by_voters <- test_total_nomiss$price_min * test_total_nomiss$voters
test_total_nomiss$price_sd_by_voters <- test_total_nomiss$price_sd * test_total_nomiss$voters

dummy_orderattribute_test <- dummies::dummy.data.frame(test_total_nomiss[,c("orderattribute2", "orderattribute3","orderattribute4")],
                                                       names=c("orderattribute2","orderattribute3","orderattribute4"), sep="_")
test_total_nomiss <- cbind(test_total_nomiss,dummy_orderattribute_test)

dim(test_total_nomiss)
holiday_test <- as.data.table(ymd(c("2015-12-08","2016-01-11","2016-02-11","2016-03-08","2016-04-10",
                                    "2016-05-10","2016-06-09","2016-07-10","2016-08-08","2016-09-08",
                                    "2016-10-13","2016-11-08","2016-12-09","2017-01-12")))

holiday_test1 <- holiday_test[,.(year=year(V1),month=month(V1),holiday_num=day(V1))]
#ggplot2::ggplot(holiday_train1,aes(x=month,y=holiday_num,colour=as.factor(year)))+geom_point()+geom_line()
test_total_nomiss <- merge(test_total_nomiss,holiday_test1,by=c("year","month"),all.x = T)
dim(test_total_nomiss)


test_total_nomiss$product_date <- ymd(paste(test_total_nomiss$year,test_total_nomiss$month,01))
test_total_nomiss$product_date_reduce_startdate <- test_total_nomiss$product_date-test_total_nomiss$startdate
test_total_nomiss$product_date_reduce_upgradedate <- test_total_nomiss$product_date-test_total_nomiss$upgradedate
test_total_nomiss$product_date_reduce_cooperatedate <- test_total_nomiss$product_date-test_total_nomiss$cooperatedate
test_total_nomiss$product_date_reduce_2014 <- test_total_nomiss$product_date-ymd("2014-01-01")
test_total_nomiss$product_date_reduce_startdate <- ifelse(test_total_nomiss$product_date_reduce_startdate<0,0,test_total_nomiss$product_date_reduce_startdate)
test_total_nomiss$product_date_reduce_upgradedate <- ifelse(test_total_nomiss$product_date_reduce_upgradedate<0,0,test_total_nomiss$product_date_reduce_upgradedate)
test_total_nomiss$product_date_reduce_cooperatedate <- ifelse(test_total_nomiss$product_date_reduce_cooperatedate<0,0,test_total_nomiss$product_date_reduce_cooperatedate)

set.seed(0)
xgb_fit <- xgboost(data = data.matrix(train_total_nomiss[,-5]),
                   label =as.numeric(train_total_nomiss$ciiquantity_month),
                   max.depth = 8, eta = 0.1,subsample=0.9,colsample_bytree=0.9,
                   min_child_weight=30,gamma=18,
                   nround =500 ,objective = "reg:linear",eval_metric="rmse")
pre_xgboost <- predict(xgb_fit ,data.matrix(test_total_nomiss))
sum(pre_xgboost<0)

#指标评价效果----------------------------------
names <- colnames(train_total_nomiss[,-5])
importance_matrix <- xgb.importance(names, model = xgb_fit)
xgb.ggplot.importance(importance_matrix)
feature_43 <- c(importance_matrix$Feature[1:40],"startdate_before201512",
                "upgradedate_before201512","cooperatedate_before201512")


#取前40个重要性特征进行xgbtree建模--------------------
xgb_fit <- xgboost(data = data.matrix(train_total_nomiss[,feature_43,with=F]),
                   label =as.numeric(train_total_nomiss$ciiquantity_month),
                   max.depth = 8, eta = 0.01,subsample=0.9,colsample_bytree=0.9,
                   min_child_weight=30,gamma=18,
                   nround =3000 ,objective = "reg:linear",eval_metric="rmse")
pre_xgboost <- predict(xgb_fit ,data.matrix(test_total_nomiss[,feature_43,with=F]))
sum(pre_xgboost<0)
sum(pre_xgboost>product_info$maxstock*30)

names <- colnames(train_total_nomiss[,feature_43,with=F])
importance_matrix <- xgb.importance(names, model = xgb_fit)
xgb.ggplot.importance(importance_matrix)

pre_xgboost_no_negative<- ifelse(pre_xgboost<0,90,pre_xgboost)
my_prediction <- prediction[,.(product_id,product_month,ciiquantity_month=round(pre_xgboost_no_negative))]
my_prediction
fwrite(my_prediction,"./model_result/model5_result.csv")



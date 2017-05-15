rm(list=ls())
library(data.table)
library(lubridate)
library(magrittr)
#读取数据---------------------------------------------------------------
product_info = fread("./data/product_info.txt")
product_quantity= fread("./data/product_quantity.txt")
prediction = fread("./data/prediction_lilei_20170320.txt")
##各列缺失值个数汇总----------------------------------------------------
for( i in names(product_info)){
  print(paste("na ",i,"=",sum(product_info[,i,with=F]==-1)))
}
for( i in names(product_quantity)){
  print(paste("na ",i,"=",sum(product_quantity[,i,with=F]==-1)))
}
##处理表1，product_info
#缺失过多的处理
col_delect <- c("railway","airport","citycenter","railway2","airport2","citycenter2",
                "district_id1","district_id2","district_id3","district_id4")
product_info[,(col_delect):=NULL]
#时间缺失的，用订单的开始时间即2014-01-01代替
product_info$startdate[product_info$startdate==-1] <- "2014-01-01"
product_info$upgradedate[product_info$upgradedate==-1] <- "2014-01-01"
product_info$cooperatedate[product_info$cooperatedate==-1] <- "2014-01-01"
product_info$startdate <- ymd(product_info$startdate)
product_info$upgradedate <- ymd(product_info$upgradedate)
product_info$cooperatedate <- ymd(product_info$cooperatedate)
#合作时间---------------------------------------------------------------------------
product_info$cooperatedate_year<- year(product_info$cooperatedate)
product_info$cooperatedate_month<- month(product_info$cooperatedate)
product_info$location <- kmeans(product_info[,.(lat,lon)],9)$cluster
#对voters做手脚
product_info[,':=' (voters_divide_maxstock = voters/maxstock,
                    voters_divide_cooperationdate_length = voters/as.numeric(ymd("2017-02-01")-product_info$cooperatedate),
                    eval_eval2 = eval - eval2,
                    eval_eval3 = eval - eval3,
                    eval_eval4 = eval - eval4,
                    eval2_eval3 = eval2 - eval3,
                    eval2_eval4 = eval2 - eval4,
                    eval3_eval4 = eval3 - eval4
)]
#订单属性1是固定的，放入product_info表中,但有缺失，缺失先用1补看看
a <- product_quantity[,.SD[c(1)], by=product_id,.SDcols =c("orderattribute1")]
product_info2 <- merge(product_info,a,by="product_id",all.x = T)
product_info2$orderattribute1[is.na(product_info2$orderattribute1)] <- 1
b <- product_quantity[,.(orderattribute2_ratio1=sum(orderattribute2==1)/.N,
                         orderattribute2_ratio2=sum(orderattribute2==2)/.N,
                         orderattribute3_ratio1=sum(orderattribute3==1)/.N,
                         orderattribute3_ratio2=sum(orderattribute3==2)/.N
),by=.(product_id)]
product_info3 <- merge(product_info2,b,by="product_id",all.x = T)
d <- product_quantity[price>-1,.(id_mean_price=mean(price,na.rm = T),
                                 id_max_price=max(price,na.rm = T),
                                 id_min_price=min(price,na.rm = T),
                                 id_sd_price=sd(price,na.rm = T)),by=product_id]
product_info4 <- merge(product_info3,d,by="product_id",all.x = T)
product_info_onehot <- product_info4

# product_info_onehot <-dummies::dummy.data.frame(product_info4,
#                                                 names=c("orderattribute1","location"),
#                                                 sep="_")%>%as.data.table()
# product_info_onehot$orderattribute1 <- product_info2$orderattribute1
product_info_onehot$orderattribute1_by_product_id <- product_info_onehot$orderattribute1*product_info_onehot$product_id
product_info_onehot <- imputeMissings::impute(product_info_onehot,method = "median/mode")%>%as.data.table()
dim(product_info_onehot)
#product_quantity价格与销量表------------------------------------------
product_quantity$ordquantity <- NULL  #因为得不到这个变量
product_quantity$product_date <- ymd(product_quantity$product_date)
product_quantity$product_date_year <- year(product_quantity$product_date)
product_quantity$product_date_month <- month(product_quantity$product_date)
product_quantity_extract <- product_quantity[,.(
  ciiquantity_month=sum(ciiquantity)
),
by=.(product_id,product_date_year,product_date_month)] %>%
  setorder(product_date_year,product_date_month)
product_quantity_extract$basetime <- as.numeric(product_quantity_extract$product_date_month)+ifelse(as.numeric(product_quantity_extract$product_date_year)==2014,0,12)

train <- merge(product_quantity_extract,product_info_onehot,by="product_id",all.x = T) %>% setorder(product_date_year,product_date_month)
dim(train)

# category_col <- c("location","orderattribute1")
train$location <- as.factor(train$location)
train$orderattribute1 <- as.factor(train$orderattribute1)

train$product_id <- as.factor(train$product_id)

#表3 prediction的处理---------------

prediction$ciiquantity_month <- NULL
prediction$product_date_year <-year(prediction$product_month) 
prediction$product_date_month<-month(prediction$product_month) 
prediction$basetime <- as.numeric(prediction$product_date_month)+ifelse(as.numeric(prediction$product_date_year)==2015,12,                                                                        ifelse(as.numeric(prediction$product_date_year)==2016,24,36))
#构建测试集
prediction_col4 <- prediction[,.(product_id,product_date_year,product_date_month,basetime)]
test <- merge(prediction_col4,product_info_onehot,by="product_id",all.x = T)%>%setorder(product_date_year,product_date_month)
dim(test)
test$location <- as.factor(test$location)
test$orderattribute1 <- as.factor(test$orderattribute1)

test$product_id <- as.factor(test$product_id)

#H2O----------------------------------------------------------------------
library(h2o)
h2o.init(nthreads = -1,max_mem_size="6g")
train <- as.h2o(train)
test <- as.h2o(test)
y <- "ciiquantity_month"
x <- setdiff(names(train), y)



my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train,
                          ntrees = 1000,
                          max_depth = 30,
                          nbins_cats = 1115,
                          col_sample_rate_per_tree=0.8,
                          # sample_rate = 0.9,
                          # nfolds = nfolds,
                          # fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)




pre_rf <- predict(my_rf ,test)
sum(pre_rf<0)
sum(pre_rf>rep(product_info$maxstock*30,14))

pre_xgboost_no_negative<- ifelse(pre_rf<0,90,pre_rf)

my_prediction <- prediction[,.(product_id,product_month,ciiquantity_month=round(as.vector(pre_xgboost_no_negative)))]
my_prediction
fwrite(my_prediction,"./model_result/model10_result.csv")



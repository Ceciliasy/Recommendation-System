#!/usr/bin/env python
from csv import reader
import sys
import pyspark
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.functions import *

new_product_ratio = 0.2
im_train_ratio = 0.8
val_ratio = 0.67

spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option","some-value").getOrCreate()
data = spark.read.json("/user/sl6486/user_dedup_aa")
data.createOrReplaceTempView("df")
#user_group = spark.sql("select reviewerID,count(*) as user_count from df group by reviewerID")
#user_group.write.format("json").save("user_group")
#product_group = spark.sql("select asin,count(*) as product_count from df group by asin")
#product_group.write.format("json").save("product_group")
user_group = spark.read.json("user_group")
user_group.createOrReplaceTempView("user_group")
product_group = spark.read.json("product_group")
product_group.createOrReplaceTempView("product_group")
# data: 200669285
# users: 5297100
# products: 4773215
n_data = df.count()
n_user = user_group.count()
n_product = product_group.count()

new_product = product_group.sample(False,new_product_ratio)
new_product.createOrReplaceTempView("new_product")
old_product = spark.sql("select df.* from df left join new_product np on df.asin=np.asin where np.product_count is null")
first_train = old_product.groupBy("reviewerID").agg({"asin": "first"})
first_train = first_train.select(col('reviewerID'), col('first(asin)').alias("asin"),col('first(asin)').alias("flag"))
first_train.createOrReplaceTempView("first_train")

omsql = "select df.* from df left join first_train on df.reviewerID = first_train.reviewerID left join new_product np on df.asin=np.asin where np.product_count is not null and first_train.flag is not null"
imsql = "select df.* from df left join first_train on df.reviewerID = first_train.reviewerID and df.asin = first_train.asin and first_train.flag is null"
trainsql = 'select df.* from df left join first_train ft on df.reviewerID = ft.reviewerID and df.asin=ft.asin and ft.flag is not null'
# im: 14370958
# om: 3591087
# train: 2162884

# TO DO
# random sample from in-matrix set, add out-of-matrix TEST set, all of these as TEST set
# split the TEST set to validation and test set
# add the rest of in-matrix to the current train data(user_count=1) set as train set
# note the train-validation-test ratio after all of these operations.make sure it is reasonable.
# Build user vocabulary and product vocabulary, map index to user_ID, save as file.
# save each dataset as rating tuples(user_id, product_id, rating)

train_ = spark.sql(trainsql)
train_.createOrReplaceTempView("train_")
im = spark.sql(imsql)
im.createOrReplaceTempView("im")
om = spark.sql(omsql)
om.createOrReplaceTempView('om')

train_im, test_im = im.randomSplit([im_train_ratio, 1-im_train_ratio], seed=1)
train = train_.union(train_im)
test_ = om.union(test_im)
val, test = test_.randomSplit([val_ratio, 1-val_ratio], seed=1)
train.createOrReplaceTempView("train")
val.createOrReplaceTempView("val")
test.createOrReplaceTempView("test")

product_id_sql = "select ID as product_id, ROW_NUMBER() over (order by (select 10)) as product_index from (select distinct(asin) as ID from train UNION select asin as ID from new_product)"
product_id = spark.sql(product_id_sql)
product_id.createOrReplaceTempView("product_id")
product_id.write.format("json").save("product_vocab")
product_id = spark.read.json("product_vocab")
user_id_sql = "select reviewerID as user_id, ROW_NUMBER() over (order by (select 10)) as user_index from train"
user_id = spark.sql(user_id_sql)
user_id.createOrReplaceTempView("user_id")
user_id.write.format("json").save("user_vocab")
user_id = spark.read.json("user_vocab")

train_tuple_sql = "select user_index, product_index, overall from train left join user_id on train.reviewerID = user_id.user_id left join product_id \
on train.asin = product_id.product_id"
train_tuple = spark.sql(train_tuple_sql)
train_tuple.createOrReplaceTempView("train_tuple")
train_tuple_list = [(i[0], i[1], i[2]) for i in train_tuple.collect()]
np.savetxt("train_tuples.txt", train_tuple_list)

val_tuple_sql = "select user_index, product_index, overall from val left join user_id on val.reviewerID = user_id.user_id left join product_id \
on val.asin = product_id.product_id"
val_tuple = spark.sql(val_tuple_sql)
val_tuple.createOrReplaceTempView("val_tuple")
val_tuple_list = [(i[0], i[1], i[2]) for i in val_tuple.collect()]
np.savetxt("val_tuples.txt", val_tuple_list)

test_tuple_sql = "select user_index, product_index, overall from test left join user_id on test.reviewerID = user_id.user_id left join product_id \
on test.asin = product_id.product_id"
test_tuple = spark.sql(test_tuple_sql)
test_tuple.createOrReplaceTempView("test_tuple")
test_tuple_list = [(i[0], i[1], i[2]) for i in test_tuple.collect()]
np.savetxt("test_tuples.txt", test_tuple_list)
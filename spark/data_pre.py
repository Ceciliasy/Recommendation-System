#!/usr/bin/env python
from csv import reader
import sys
import pyspark
from pyspark.sql import SparkSession

new_product_ratio = 0.2


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
sql = "select count(*) from df where reviewerID in (select reviewerID from user_group where user_count>1"
trainsql = "select * from (select df.*,np.product_count as product_count from df left join new_product np on df.asin = np.asin) t, user_group where t.reviewerID = user_group.reviewerID and user_group.user_count == 1 and t.product_count is null"
omsql = "select * from df,user_group u,new_product np where df.reviewerID = u.reviewerID and df.asin = np.asin and u.user_count > 1"
imsql = "select * from (select df.*,np.product_count as product_count from df left join new_product np on df.asin = np.asin) t, user_group where t.reviewerID = user_group.reviewerID and user_group.user_count > 1 and t.product_count is null"
n_train = spark.sql(trainsql).count()
n_om = spark.sql(omsql).count()
n_im = spark.sql(imsql).count()
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




#result.write.save("task1-sql.out",format="text")
#result.show()


# Databricks notebook source
from fbprophet import Prophet
import logging
import pandas as pd
from pyspark.sql.types import *

# disable informational messages from fbprophet
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

df = spark.read.load("/mnt/db-stage-lake/jagout/m_transactions_parquet", inferSchema=True)
df.createOrReplaceTempView("df")
income = spark.read.load("/mnt/db-stage-lake/jagout/m_Feb21_kmeans_credits_12m", inferSchema=True)
income.createOrReplaceTempView("income")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct b.CID) from income b where b.TRANS_SUM_6M_SUM_sal_cr > 1500

# COMMAND ----------

# MAGIC %sql
# MAGIC --create table if not exists cid_history_val using parquet partitioned by (cid) location '/mnt/db-stage-lake/jagout/test_cid_hist'
# MAGIC --AS
# MAGIC --select a.CID as cid, to_date(concat(a.YEARMONTH,'-01'), 'yyyy-MM-dd') as ds, SUM(a.AMT) as y 
# MAGIC --from df a inner join income b on a.CID = b.CID 
# MAGIC -- and b.TRANS_SUM_6M_SUM_sal_cr > 1500 and a.TRN = 'Salary Payments'
# MAGIC --group by a.CID, a.yearmonth 
# MAGIC --order by cid, ds;
# MAGIC 
# MAGIC select a.CID as cid, to_date(concat(a.YEARMONTH,'-01'), 'yyyy-MM-dd') as ds, SUM(a.AMT) as y 
# MAGIC from df a 
# MAGIC where a.CID in (
# MAGIC     select distinct b.CID from income b where b.TRANS_SUM_6M_SUM_sal_cr > 1500 and cid >= 4415 limit 100
# MAGIC ) 
# MAGIC and a.TRN = 'Salary Payments'
# MAGIC group by a.CID, a.yearmonth 
# MAGIC order by cid, ds;

# COMMAND ----------

sql_statement = '''
  select a.CID as cid, to_date(concat(a.YEARMONTH,'-01'), 'yyyy-MM-dd') as ds, SUM(a.AMT) as y 
  from df a 
  where a.CID in (
      select distinct b.CID from income b where b.TRANS_SUM_6M_SUM_sal_cr > 1500
  ) 
  and a.TRN = 'Salary Payments'
  group by a.CID, a.yearmonth 
  --order by cid, ds
  '''

cid_amt_history = (
  spark
    .sql( sql_statement )
    #.repartition(sc.defaultParallelism, ['CID'])
    .repartition(10000, ['CID'])
  ).cache()

# COMMAND ----------

cid_amt_history.count()

# COMMAND ----------

result_schema = StructType([
  StructField('ds',DateType()),
  StructField('cid',IntegerType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

# COMMAND ----------

def forecast_cid_amt( history_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # TRAIN MODEL AS BEFORE
  # --------------------------------------
  # remove missing values (more likely at day-store-item level)
  history_pd = history_pd.dropna()
  
  # configure the model
  model = Prophet(
    interval_width=0.10,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    seasonality_mode='additive'
    )
  
  # train the model
  model.fit( history_pd )
  # --------------------------------------
  
  # BUILD FORECAST AS BEFORE
  # --------------------------------------
  # make predictions
  future_pd = model.make_future_dataframe(
    periods=12, 
    freq='M', 
    include_history=True
    )
  forecast_pd = model.predict( future_pd )  
  # --------------------------------------
  
  # ASSEMBLE EXPECTED RESULT SET
  # --------------------------------------
  # get relevant fields from forecast
  f_pd = forecast_pd[ ['ds', 'yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  # get relevant fields from history
  h_pd = history_pd[['ds','cid','y']].set_index('ds')
  
  # join history and forecast
  results_pd = f_pd.join( h_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)
  
  # get store & item from incoming data set
  results_pd['cid'] = history_pd['cid'].iloc[0]
  # --------------------------------------
  
  # return expected dataset
  return results_pd[ ['ds', 'cid', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]  

# COMMAND ----------

from pyspark.sql.functions import current_date

results = (
  cid_amt_history
    .groupBy('cid')
      .applyInPandas(forecast_cid_amt, schema=result_schema)
    .withColumn('training_date', current_date() )
    )

results.createOrReplaceTempView('new_forecasts')

display(results)

# COMMAND ----------

display(results.where("cid == 4415").orderBy("ds"))

# COMMAND ----------

results.write.partitionBy("ds").mode("append").parquet("/mnt/db-stage-lake/jagout/m_transactionsforecasts_parquet")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- create forecast table
# MAGIC create table if not exists forecasts (
# MAGIC   date date,
# MAGIC   cid integer,
# MAGIC   amt float,
# MAGIC   amt_predicted float,
# MAGIC   amt_predicted_upper float,
# MAGIC   amt_predicted_lower float,
# MAGIC   training_date date
# MAGIC   )
# MAGIC using parquet location '/mnt/db-stage-lake/jagout/forecasts'
# MAGIC partitioned by (date);
# MAGIC 
# MAGIC -- load data to it
# MAGIC insert into forecasts
# MAGIC select 
# MAGIC   ds as date,
# MAGIC   cid,
# MAGIC   y as amt,
# MAGIC   yhat as amt_predicted,
# MAGIC   yhat_upper as amt_predicted_upper,
# MAGIC   yhat_lower as amt_predicted_lower,
# MAGIC   training_date
# MAGIC from new_forecasts;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from forecasts limit 100

# COMMAND ----------

display(results.where("cid == 4415").orderBy("ds"))

# COMMAND ----------

sc.defaultParallelism

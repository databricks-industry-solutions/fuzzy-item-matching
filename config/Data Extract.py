# Databricks notebook source
# MAGIC %sh 
# MAGIC rm -rf /dbfs/tmp/matching/
# MAGIC mkdir /dbfs/tmp/matching/
# MAGIC cd /dbfs/tmp/matching/
# MAGIC wget https://dbs.uni-leipzig.de/file/Abt-Buy.zip
# MAGIC unzip -o Abt-Buy.zip -d bronze/

# COMMAND ----------

# MAGIC %sh ls /dbfs/tmp/matching/

# COMMAND ----------



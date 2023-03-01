# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/fuzzy-item-matching. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/product-matching-with-ml.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to build a set of product-pairs representing potential matches between products sold by two retailers. 

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans

import pyspark.sql.functions as f
from pyspark.sql.window import Window

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# DBTITLE 1,Set up mlflow experiment in the user's personal workspace folder
model_name = 'abt_buy_pipelines'
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/fuzzy_item_matching"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# MAGIC %md ## Step 1: Retrieve Relevant Product Features
# MAGIC 
# MAGIC Our goal is now to determine products in the Abt dataset *likely* to be matches for products in the Buy dataset.   We will use the likely matches as the input for later model training, forcing our model to discern between products likely to match and those actually matching.  For the generation of this dataset, we'll limit our analysis to product names and descriptions:

# COMMAND ----------

# DBTITLE 1,Retrieve Relevant Features
features = (
  spark
    .table('DELTA.`/tmp/matching/silver/abtbuyfeatures`')
    .selectExpr('table', 'id', 'name_features', 'descript_features')
  )

display(features.limit(3))

# COMMAND ----------

# MAGIC 
# MAGIC %md ## Step 2: Generate Candidate Pairs
# MAGIC 
# MAGIC In order to compare products in the Abt dataset to those in the Buy dataset, we could produce an exhaustive set of comparisons between the two.  With about 1,100 products in each, this would require about 1.2-million (1,100 * 1,100) product pairs to be generated. For small datasets such as these, this may be an effective strategy, but for anything sizeable, this quickly creates a scaling challenge. To overcome this problem, we will make use of [Locality Sensitive Hashing (LSH)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing).
# MAGIC 
# MAGIC LSH is a fairly simple technique which divides high-dimensional spaces using randomly generated hyperplanes.  Those items above a hyperplane are expected to be more similar to each other than those items below the hyperplane. Using multiple hyperplanes, the dimensional space is divided into buckets of relatively similar items.  By reinitializing and repeating this process, several sets of these buckets are generated within which items truly more similar to each other than to other items will more likely be placed in at least one bucket together.  This overcomes the problems with the random nature of the hyperplanes but still keeps the list of likely similar items for any one item reasonably low:
# MAGIC 
# MAGIC **NOTE** For a deeper dive into LSH, please review the notebooks associated with our [recommender blog post](https://databricks.com/blog/2020/12/18/personalizing-the-customer-experience-with-recommendations.html).
# MAGIC 
# MAGIC **NOTE** MinHashLSH is also commonly used in NLP applications. With that algorithm, the Jaccard similarity between objects is compared.  The input vectors are typically set to 0 or 1 indicating the presence of a value in a set.  If using MinHashLSH, be sure to exclude high-frequency n-grams as their low IDF scores will still be interpreted as indicators of the presence of the n-gram and will cause them to be given equal consideration with the others.

# COMMAND ----------

# DBTITLE 1,Bucket Products on Name
# divide names into buckets
name_lsher = BucketedRandomProjectionLSH(
    inputCol='name_features', 
    outputCol='name_hashes', 
    bucketLength=1.0,
    numHashTables=10
    ).fit(features )

# display results
display( 
  name_lsher
    .transform(features)
    .select('table', 'id', 'name_hashes')
    .limit(3)
  )

# COMMAND ----------

# MAGIC %md With our pipeline assembled, we can now use it to find similar products.  We'll fit the model using our combined Abt and Buy datasets and then perform an approximate similarity join between these to find likely matches:

# COMMAND ----------

# DBTITLE 1,Retrieve Candidate Pairs Based on Name
# transform abt and buy datasets separately
names_hashed = name_lsher.transform(features) 

# set to max so that any items in a shared LSH bucket are considered candidates
max_distance_from_target = 1.3
 
# calculate distances between products
name_candidates = (
    name_lsher.approxSimilarityJoin(
      names_hashed.filter(f.expr("table='abt'")),
      names_hashed.filter(f.expr("table='buy'")),  
      threshold = max_distance_from_target, 
      distCol='distance'
      )
    ).selectExpr(
      'datasetA.table as tableA',
      'datasetA.id as idA',
      'datasetB.table as tableB',
      'datasetB.id as idB',
      'distance as name_distance'
      )

display(
  name_candidates.limit(5)
  )

# COMMAND ----------

# DBTITLE 1,Count Candidate Pairs
name_candidates.count()

# COMMAND ----------

# MAGIC %md We now have a list of candidate pairs based on similarities between names.  The number of candidate pairs generated are a bit smaller than the 1.2-million we might have considered, and this is a count we can adjust by tuning our LSH model or shortening the distance used in the approximate similarity join.
# MAGIC 
# MAGIC Let's now repeat this for description:
# MAGIC 
# MAGIC **NOTE** We are using a slightly more restrictive max distance value with our descriptions in order to better constrain our data.

# COMMAND ----------

# DBTITLE 1,Bucket Products on Description
# divide products into buckets of likely-similar items
descript_lsher = (
  BucketedRandomProjectionLSH(
    inputCol='descript_features', 
    outputCol='descript_hashes', 
    bucketLength=1.0,
    numHashTables=10
    )
  ).fit(features)

# set max distance to allow in results
max_distance_from_target = 1.1

# calculate distances between products
description_candidates = (
    descript_lsher.approxSimilarityJoin(
      descript_lsher.transform( features.filter(f.expr("table='abt'")) ),
      descript_lsher.transform( features.filter(f.expr("table='abt'")) ),
      threshold = max_distance_from_target, 
      distCol='distance'
      )
    ).selectExpr(
      'datasetA.table as tableA', 
      'datasetA.id as idA', 
      'datasetB.table as tableB', 
      'datasetB.id as idB', 
      'distance as description_distance'
      )

# show results
display(
  description_candidates.limit(5)
  )

# COMMAND ----------

# DBTITLE 1,Count Candidate Pairs
description_candidates.count()

# COMMAND ----------

# MAGIC %md Before moving on, we should preserve our LSH models for later re-use.  As with our pipelines, these are not standard models and so must be persisted in mlflow as artifacts:

# COMMAND ----------

model_name = 'abt_buy_lsh'

# persist pipeline for later re-user
with mlflow.start_run(run_name=model_name) as run:
  
  # persist each pipeline object as mlflow artifact
  lshmodels = {'name_lsher':None, 'descript_lsher':None}
  for lshmodel in lshmodels:
    
    # define local path for temp save
    lshmodel_path = '/tmp/{0}'.format(lshmodel)
    
    # save pipeline to temp location
    eval(lshmodel).write().overwrite().save(lshmodel_path)
    
    # persist to mlflow
    mlflow.log_artifact('/dbfs'+lshmodel_path)

# COMMAND ----------

# MAGIC %md ## Step 3: Merge & Label Candidate Pairs
# MAGIC 
# MAGIC We now have a set of candidate pairs based on names and another based on descriptions.  Let's combine these two using a simple union.  We will also append our expert-matched pairs to ensure our candidate list is complete.
# MAGIC 
# MAGIC Please note that **union()** and **unionAll()** in pyspark are equivalent to a **UNION ALL** in SQL.  For this reason, we'll need to add a **distinct()** after the union operations:

# COMMAND ----------

# DBTITLE 1,Retrieve Known Matches
matches = (
  spark
    .table('DELTA.`/tmp/matching/silver/abtbuy_matched`')
    .selectExpr('idAbt as idA', 'idBuy as idB')
  )

display(matches)

# COMMAND ----------

# DBTITLE 1,Assemble Consolidated Set of Candidate Pairs
candidate_pairs = (
  matches
  .union(name_candidates.select('idA', 'idB'))
  .union(description_candidates.select('idA', 'idB'))
  .distinct()
  )

display(candidate_pairs)

# COMMAND ----------

# DBTITLE 1,Count Candidate Pairs
candidate_pairs.count()

# COMMAND ----------

# MAGIC %md We can now label those pairs as to whether they are matches or not:

# COMMAND ----------

# DBTITLE 1,Assign Labels to Candidate Pairs
labeled_pairs = (
  candidate_pairs
    .join(
      matches.withColumn('label',f.lit(1)),
      on=['idA', 'idB'],
      how='leftouter'
      )
    .withColumn('label', f.expr('coalesce(label,0)'))
  )

display(labeled_pairs)

# COMMAND ----------

# DBTITLE 1,Persist Labeled Pairs
(
  labeled_pairs
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/matching/silver/labeledpairs')
)

# COMMAND ----------

# MAGIC %md But before moving on, we should take a look at the number of matched and unmatched values in our pairs set:

# COMMAND ----------

# DBTITLE 1,Calculate Proportion of Labels in Set
class_ratios = (
  labeled_pairs
    .groupBy('label')
      .agg(f.count('*').alias('instances'))
    .withColumn('total_instances', f.sum('instances').over(Window.partitionBy()))
    .withColumn('ratio', f.expr('(instances)/total_instances'))
    )

display( class_ratios )

# COMMAND ----------

# MAGIC %md The class imbalance is to be expected.  We have many products that could be matches for one another and only a small number that actually are.  When we perform our model training, we'll need to be sure to consider this issue.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library / data source                  | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Abt-Buy                                | dataset                 | CC-BY 4.0  | https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution  |

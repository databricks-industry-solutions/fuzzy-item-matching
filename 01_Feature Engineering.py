# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/fuzzy-item-matching. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/product-matching-with-ml.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is access the data and generate the features used in later notebooks.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as f

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram, SQLTransformer, HashingTF, IDF, Word2Vec, Normalizer, Imputer, VectorAssembler
from pyspark.ml import Pipeline

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.cluster import KMeans
import numpy as np

# COMMAND ----------

# DBTITLE 1,Set up mlflow experiment in the user's personal workspace folder
model_name = 'abt_buy_pipelines'
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/fuzzy_item_matching"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# MAGIC %md ## Step 1: Access the Dataset
# MAGIC 
# MAGIC To demonstrate entity-resolution techniques, we will make use of the Abt-Buy dataset, which identifies products sold by online two companies, *i.e.* abt.com and buy.com, and provides a *golden-record* dataset matching products from both. This dataset (and many others) is made available by the [Database Group Leipzig](https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution) for the purpose of  evaluating entity-resolution techniques.  
# MAGIC 
# MAGIC Having downloaded the data from the website and uploaded the unzipped data files to the appropriate folders under a [mount point](https://docs.databricks.com/data/databricks-file-system.html) named */mnt/matching*. 
# MAGIC 
# MAGIC For illustration, we have automated this step and downloaded the data in a temporary folder */tmp/matching/*. 

# COMMAND ----------

# MAGIC %run "./config/Data Extract"

# COMMAND ----------

# MAGIC %md We can make the data accessible as follows:

# COMMAND ----------

# DBTITLE 1,Abt.com Products
# schema for abt dataset
abt_schema = StructType([
  StructField('id', IntegerType()),
  StructField('name', StringType()),
  StructField('description', StringType()),
  StructField('price', StringType())
  ])

# read data and write to delta
(
  spark
    .read
      .csv('/tmp/matching/bronze/Abt.csv', sep=',', header=True, schema=abt_schema)
      .withColumn('price', f.expr("cast(replace(price, '$', '') as float)"))  # strip characters from numerical strings and convert to float
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema', 'true')
      .save('/tmp/matching/silver/abt')
  )

# view data
display( spark.table('DELTA.`/tmp/matching/silver/abt`') )

# COMMAND ----------

# DBTITLE 1,Buy.com Products
# schema for buy dataset
buy_schema = StructType([
  StructField('id', IntegerType()),
  StructField('name', StringType()),
  StructField('description', StringType()),
  StructField('manufacturer', StringType()),
  StructField('price', StringType())
])

# read data and write to delta
(
  spark
    .read
      .csv('/tmp/matching/bronze/Buy.csv', sep=',', header=True, schema=buy_schema)
      .withColumn('price', f.expr("cast(replace(price, '$', '') as float)")) # strip characters from numerical strings and convert to float
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema', 'true')
      .save('/tmp/matching/silver/buy')
  )

# view data
display( spark.table('DELTA.`/tmp/matching/silver/buy`') )

# COMMAND ----------

# MAGIC %md As with most datasets used for entity-resolution, a set of products representing expert-determined matches is provided:

# COMMAND ----------

# DBTITLE 1,Expert-Matched Data
# schema for buy dataset
abtbuy_matched_schema = StructType([
   StructField('idAbt', IntegerType()),
   StructField('idBuy', IntegerType())
  ])

# read data and write to delta
(
  spark
    .read
      .csv('/tmp/matching/bronze/abt_buy_perfectMapping.csv', sep=',', header=True, schema=abtbuy_matched_schema)
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema', 'true')
      .save('/tmp/matching/silver/abtbuy_matched')
  )

# view data
display( spark.table('DELTA.`/tmp/matching/silver/abtbuy_matched`') )

# COMMAND ----------

# MAGIC %md ## Step 2: Create Features
# MAGIC 
# MAGIC Products from Abt and Buy will be compared based on names, descriptions and prices.  In the following steps, we will generate features for each of these attributes leveraging the unioned set of products from each set:

# COMMAND ----------

# DBTITLE 1,Combine Products for Feature Engineering
# relevant attributes from abt dataset
abt = (
  spark
    .table('DELTA.`/tmp/matching/silver/abt`')
    .withColumn('table', f.lit('abt')) # identify source of records
    .selectExpr('table', 'id', 'name', "COALESCE(description, '') as description", 'price')
  )

# relevant attributes from buy dataset
buy = (
  spark
    .table('DELTA.`/tmp/matching/silver/buy`')
    .withColumn('table', f.lit('buy')) # identify source of records
    .selectExpr('table', 'id', 'name', "COALESCE(description, '') as description", 'price')
  )

# union the sets together
all = abt.unionAll(buy)

display(all)

# COMMAND ----------

# MAGIC %md ### Step 2a: Name Features 
# MAGIC 
# MAGIC Names often provide a meaningful basis for comparisons but they can also include information that helps differentiate various similar offerings from the same vendor.  For example, you may be attempting to purchase a specific type of dishwasher and find it listed as *Frigidaire 24' White Built-In Dishwasher*.  But it could also be listed as *Fridgidaire 24' BuiltIn Dish Washer (White)*. While these minor variations are relatively easy for us as humans to sort through, for software, the problem requires a bit of creativity.
# MAGIC 
# MAGIC The initial sequence of steps we will employ to enable name comparisons will be:</p>
# MAGIC 
# MAGIC 1. Split names into individual words, *aka* [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)
# MAGIC 2. Remove any common [stop words](https://en.wikipedia.org/wiki/Stop_word) (like *the*, *and*, of, *etc.*)
# MAGIC 3. Parse words into overlapping, three-character sequences, *aka* [n-grams](https://en.wikipedia.org/wiki/N-gram#:~:text=In%20the%20fields%20of%20computational,a%20text%20or%20speech%20corpus) or [shingles](https://en.wikipedia.org/wiki/W-shingling)
# MAGIC 
# MAGIC There's plenty of room for varying these steps and adding others, but our goal is a set of words or word-chunks that are representative of the name string.
# MAGIC 
# MAGIC The parsing of words into character-based n-grams will require us to explode the bag-of-words associated with each product name, split individual words into character-arrays, and then assemble the n-character combinations found in each word.  We will then need to collapse those n-grams back into a single array representing all the character-based n-grams found within a product name:
# MAGIC 
# MAGIC **NOTE** We are limiting displayed results to three records to reduce the size of this notebook.  This does not affect the business logic, only the output displayed in the notebook.

# COMMAND ----------

# DBTITLE 1,Split Product Names into N-Grams
# split name into word-tokens
name_tokenizer = (
  RegexTokenizer(
    minTokenLength=1, 
    pattern='[^\w\d]',  # split on non-words and non-digits (eliminates punctuation)
    inputCol='name', 
    outputCol='name_tokens',
    toLowercase=True
    )
  )

# remove stop words from word-tokens
name_filterer = (
  StopWordsRemover(
    inputCol=name_tokenizer.getOutputCol(), 
    outputCol='name_filtered'
    )
  )

# transform each word-token into a character array
name_melter_OutputCol = 'name_word_array'
name_melter = (
  SQLTransformer(
    statement="""
      SELECT
        x.*,
        y.word,
        split(y.word,'') as {0} 
      FROM __THIS__ x 
      LATERAL VIEW explode(x.{1}) y as word
      """.format(name_melter_OutputCol, name_filterer.getOutputCol())
    )
  )

# n-gram the characters of each word-token (char array)
name_ngramer = (
  NGram(
    n=3, 
    inputCol=name_melter_OutputCol, 
    outputCol='name_ng'
    )
  )

# collapse n-grams of characters into a single list for this item
name_collapser_OutputCol = 'name_ngs'
name_collapser = (
  SQLTransformer(
    statement="""
      SELECT 
        x.table, 
        x.id, 
        x.name, 
        x.description,
        x.price,
        collect_list(y.ng) as {0} 
      FROM __THIS__ x 
      LATERAL VIEW explode(x.{1}) y as ng 
      GROUP BY 
        x.table, 
        x.id, 
        x.name,
        x.description,
        x.price
        """.format(name_collapser_OutputCol, name_ngramer.getOutputCol())
    )
  )

# assemble pipeline
name_pipeline = Pipeline(stages=[name_tokenizer, name_filterer, name_melter, name_ngramer, name_collapser]).fit(all)

# display results
display( 
  name_pipeline
    .transform(all)
    .limit(3)
  )

# COMMAND ----------

# MAGIC %md With names re-organized as n-grams, we can now calculate a score for each based on its occurrence in both the individual product name as well as the overall set of product names in the dataset.  This is done through a simple [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) scoring mechanism which we will normalize using an [L2-normalization](https://machinelearningmastery.com/vector-norms-machine-learning/) as is standard for most TF-IDF scoring scenarios.
# MAGIC 
# MAGIC Here, we are implementing the term-frequency (TF) portion of the TF-IDF calculation with the *binary* argument set to *True*. This forces the TF value to be 1 if an n-gram is present.  In effect, this causes our TF-IDF scores to really be IDF-only scores.   
# MAGIC 
# MAGIC One last thing about the IDF calculation that's worth noting is that is lowers the value for n-grams as their frequency in the overall collection of n-grams increases (hence the *inverse* in inverse document frequency). As such, very common n-grams will receive little consideration in later evaluations but they will still take up space in your feature vector.  If you would like to exclude the most common n-grams, you might consider a function to remove n-grams from the dataset based on some relative threshold using something like a SQL Transformer. This isn't implemented here for the sake of simplicity:
# MAGIC 
# MAGIC **NOTE** For a more detailed examination of TF-IDF scoring and L2-normalization, please review the notebooks associated with our [recommender blog post](https://databricks.com/blog/2020/12/18/personalizing-the-customer-experience-with-recommendations.html).

# COMMAND ----------

# DBTITLE 1,Calculate TF-IDF Scores for N-Grams
# construct vector for ngrams
name_tfer = (
   HashingTF(
     inputCol=name_collapser_OutputCol, 
     outputCol='name_ngtf', 
     numFeatures=pow(36,3),
     binary=True  # we don't care about how frequently this pops up in the specific name, only in the overall name corpus
     )
  )

## calculate inverse document frequency for ngrams
name_idfer = (
  IDF(
    inputCol=name_tfer.getOutputCol(), 
    outputCol='name_ngtfidf'
    )
  )

# normalize tf-idf scores
name_normalizer = (
  Normalizer(inputCol=name_idfer.getOutputCol(), outputCol='name_features', p=2.0)
  )

# assemble pipeline
name_pipeline = Pipeline(stages=[name_tokenizer, name_filterer, name_melter, name_ngramer, name_collapser, name_tfer, name_idfer, name_normalizer]).fit(all)

# generate features from name
name_features = name_pipeline.transform(all)

# display results
display( 
  name_features
    .select('table', 'id', 'name', name_collapser_OutputCol, name_idfer.getOutputCol(), name_normalizer.getOutputCol())
    .limit(3)
  )

# COMMAND ----------

# MAGIC %md ### Step 2b: Description Features
# MAGIC 
# MAGIC We rarely have just the name to go off of for determining a match.  Information such as a product's description can help us refine our comparison of products, distinguishing similarly named but otherwise different products.
# MAGIC 
# MAGIC The steps we take to calculate the distances between product descriptions are very similiar to those we might take with product name except that we might examine the lengthier product descriptions in terms of content similarity as opposed to the word-construction comparisons supported by the character-based n-grams.  One approach for this is simply to calculate a word-based n-gram for each product:
# MAGIC 
# MAGIC **NOTE** Because there may be NULL descriptions in the dataset, we replaced NULL values with empty strings in an earlier step.

# COMMAND ----------

# DBTITLE 1,Process Words in Product Description
# split name into word-tokens
descript_tokenizer = (
  RegexTokenizer(
    minTokenLength=1, 
    pattern='[^\w\d]',  # split on non-words and non-digits (eliminates punctuation)
    inputCol='description', 
    outputCol='descript_tokens',
    toLowercase=True
    )
  )

# remove stop words from word-tokens
descript_filterer = (
  StopWordsRemover(
    inputCol=descript_tokenizer.getOutputCol(), 
    outputCol='descript_filtered'
    )
  )

# n-gram the characters of each word-token (char array)
descript_ngramer = (
  NGram(
    n=2, 
    inputCol=descript_filterer.getOutputCol(), 
    outputCol='descript_ng'
    )
  )

# assemble pipeline
descript_pipeline = Pipeline(stages=[descript_tokenizer, descript_filterer, descript_ngramer]).fit(all)

# display results
display( 
  descript_pipeline
    .transform(all) 
    .select( 'table', 'id', 'description', descript_ngramer.getOutputCol())
    .limit(3)
  )

# COMMAND ----------

# DBTITLE 1,Calculate TF-IDF Scores
# construct vector for ngrams
descript_tfer = (
   HashingTF(
     inputCol=descript_ngramer.getOutputCol(), 
     outputCol='descript_ngtf', 
     binary=False
     )
  )

# calculate inverse document frequency for ngrams
descript_idfer = (
  IDF(
    inputCol=descript_tfer.getOutputCol(), 
    outputCol='descript_ngtfidf'
    )
  )

# normalize tf-idf scores
descript_normalizer = (
  Normalizer(inputCol=descript_idfer.getOutputCol(), outputCol='descript_features', p=2.0)
  )

# assemble pipeline
descript_pipeline = Pipeline(stages=[descript_tokenizer, descript_filterer, descript_ngramer, descript_tfer, descript_idfer, descript_normalizer]).fit(all)

# transform data
descript_features = descript_pipeline.transform(all)

# display results
display( 
  descript_features
    .select('table', 'id', 'description', descript_ngramer.getOutputCol(), descript_idfer.getOutputCol())
    .limit(3)
  )

# COMMAND ----------

# MAGIC %md The n-gram technique used as the basis of description scoring examines how words occur in pairs.  A slightly more sophisticated variation on this, known as [Word2Vec](https://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec), examines how words are determined by preceding words.  The scores generated by Word2Vec are known as *embeddings*.  They are essentially the weights trained for a hidden layer in a neural network.  As such, their interpretability can be challenging but they are very useful in examining word relationships within longer text:

# COMMAND ----------

# DBTITLE 1,Leverage Word2Vec in Pipeline
# split name into word-tokens
descript_tokenizer = (
  RegexTokenizer(
    minTokenLength=1, 
    pattern='[^\w\d]',  # split on non-words and non-digits (eliminates punctuation)
    inputCol='description', 
    outputCol='descript_tokens',
    toLowercase=True
    )
  )

# remove stop words from word-tokens
descript_filterer = (
  StopWordsRemover(
    inputCol=descript_tokenizer.getOutputCol(), 
    outputCol='descript_filtered'
    )
  )

# calculate word2vec embeddings
descript_w2ver = (
  Word2Vec(
    inputCol=descript_filterer.getOutputCol(),
    outputCol='descript_embeddings',
    vectorSize=50,
    minCount=3,
    maxSentenceLength=1000,
    maxIter=100
    )
  )

# normalize tf-idf scores
descript_normalizer = (
  Normalizer(inputCol=descript_w2ver.getOutputCol(), outputCol='descript_features', p=2.0)
  )

# assemble pipeline
descript_pipeline = Pipeline(stages=[descript_tokenizer, descript_filterer, descript_w2ver, descript_normalizer]).fit(all)

# transform data
descript_features = descript_pipeline.transform(all)

display( 
  descript_features
    .select('table', 'id', 'description', descript_normalizer.getOutputCol())
    .limit(3) 
  )

# COMMAND ----------

# MAGIC %md ### Step 2c: Price Features
# MAGIC 
# MAGIC The last attribute we will consider is price. It's reasonable to think that two sites selling the same product should offer similar (but not necessarily identical) prices. 
# MAGIC 
# MAGIC Price is a fairly straightforward feature, and as such, the only thing we need to do is convert the scalar prices into the vector representations our later steps require:

# COMMAND ----------

# DBTITLE 1,Transform Price to Features
# assemble pricing features
price_vectorizer = VectorAssembler(inputCols=['price'], outputCol='price_features', handleInvalid='keep')

# assemble pipeline
price_pipeline = Pipeline(stages=[price_vectorizer]).fit(all)

# apply pipeline to generate features
price_features = price_pipeline.transform(all)

display(
  price_features
  .select('table','id','price',price_vectorizer.getOutputCol())
  .limit(5))

# COMMAND ----------

# MAGIC %md ## Step 3: Persist Features & Pipelines
# MAGIC 
# MAGIC We now have features derived from each product's name, description and price.  We will persist these data for use in later steps:

# COMMAND ----------

# DBTITLE 1,Assemble All Product Features
features = (
  name_features
    .join(descript_features, on=['table','id'])
    .join(price_features, on=['table','id'])
    .select('table','id','name_features','descript_features','price_features')
  )

display(features.limit(3))

# COMMAND ----------

# DBTITLE 1,Persist Features for ReUse
(
  features
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .save('/tmp/matching/silver/abtbuyfeatures')
    )

# COMMAND ----------

# MAGIC %md We might also persist our pipelines for re-use.  [Mlflow](https://mlflow.org/) provides a very convenient mechanism for persisting models, but because our pipelines do not contain actual predictive models, just transformation steps, we must persist these as generic [artifacts](https://www.mlflow.org/docs/latest/tracking.html#concepts) associated with an experiment run:

# COMMAND ----------

# DBTITLE 1,Persist Pipelines for ReUse
# persist pipeline for later re-user
with mlflow.start_run(run_name=model_name) as run:
  
  # persist each pipeline object as mlflow artifact
  pipelines = {'name_pipeline':None, 'descript_pipeline':None, 'price_pipeline':None}
  for pipeline in pipelines:
    
    # define local path for temp save
    pipeline_path = '/tmp/{0}'.format(pipeline)
    
    # save pipeline to temp location
    eval(pipeline).write().overwrite().save(pipeline_path)
    
    # persist to mlflow
    mlflow.log_artifact('/dbfs'+pipeline_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library / data source                  | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Abt-Buy                                | dataset                 | CC-BY 4.0  | https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution  |

# COMMAND ----------



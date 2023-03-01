# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/fuzzy-item-matching. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/product-matching-with-ml.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is use the product pairs assembled in the prior notebook to train and deploy a model capable of identifying potential product matches between two retailers.  

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pandas as pd
import numpy as np

import pyspark.sql.functions as f
from pyspark.sql.types import *

from pyspark.ml import PipelineModel
from pyspark.ml.feature import BucketedRandomProjectionLSHModel

from delta.tables import *

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

import time

# COMMAND ----------

# DBTITLE 1,Set up mlflow experiment in the user's personal workspace folder
model_name = 'abt_buy_pipelines'
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/fuzzy_item_matching"
experiment = mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# MAGIC %md ## Step 1: Prepare Pairs Data
# MAGIC 
# MAGIC Let's get started by retrieving the labeled pairs assembled in the last notebook: 

# COMMAND ----------

# DBTITLE 1,Retrieve Labeled Pairs
labeled_pairs = (
  spark.table('DELTA.`/tmp/matching/silver/labeledpairs`')
  )

display(labeled_pairs)

# COMMAND ----------

# MAGIC %md For each product in the pair, we can now attach features:

# COMMAND ----------

# DBTITLE 1,Assign Features to Products in Pairs
features = (
  spark.table('DELTA.`/tmp/matching/silver/abtbuyfeatures`')
  )

labeled_features = (
  labeled_pairs
    .join(features.filter(f.expr("table='abt'")).alias('A'), on=f.expr('idA=A.id'))
    .join(features.filter(f.expr("table='buy'")).alias('B'), on=f.expr('idB=B.id'))
    .selectExpr('label', 'idA', 'A.name_features', 'A.descript_features', 'A.price_features', 'idB', 'B.name_features', 'B.descript_features', 'B.price_features')
)

display(labeled_features.limit(3))

# COMMAND ----------

# MAGIC %md The features retrieved for each product provide numerical representations of its name, description and price attributes. The similarity between these products can now be calculated in terms of the distance between these numerical representations.  While there are many ways to do this, we might simply calculate a squared Euclidean distance between each attribute. This is most easily addressed using Scala which has native support for the vector type use for each feature:

# COMMAND ----------

# DBTITLE 1,Define Function for Squared Distance Calculations
# MAGIC %scala
# MAGIC  
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors}
# MAGIC  
# MAGIC // from https://github.com/karlhigley/spark-neighbors/blob/master/src/main/scala/com/github/karlhigley/spark/neighbors/linalg/DistanceMeasure.scala
# MAGIC val squared_dist = udf { (v1: Vector, v2: Vector) =>
# MAGIC  Vectors.sqdist(v1, v2)
# MAGIC }
# MAGIC  
# MAGIC spark.udf.register("squared_dist", squared_dist)

# COMMAND ----------

# DBTITLE 1,Calculate Distances between Attributes
labeled_distances = (
    labeled_features
      .withColumn('name_sqdist', f.expr('squared_dist(A.name_features, B.name_features)'))
      .withColumn('descript_sqdist', f.expr('squared_dist(A.descript_features, B.descript_features)'))
      .withColumn('price_sqdist', f.expr('squared_dist(A.price_features, B.price_features)'))
      .select('idA', 'idB', 'name_sqdist', 'descript_sqdist', 'price_sqdist', 'label')
    )

display(
  labeled_distances
  )

# COMMAND ----------

# MAGIC %md We now have our training features, the squared distance between the names, the descriptions, and the price for each product pair.  Because the size of the feature set is more manageable than before, we can export our data to pandas, allowing us to make use of a wider array of machine learning algorithms later:

# COMMAND ----------

# DBTITLE 1,Convert to Pandas
data_pd = labeled_distances.toPandas()
data_pd

# COMMAND ----------

# MAGIC %md To complete our data prep, we split our data into training, validation and testing set. Notice that we are stratifying on the label to ensure we receive a similar proportion of matched items in each set:

# COMMAND ----------

# DBTITLE 1,Split into Train, Validate & Test Sets
X = data_pd.drop(['label'],axis=1).values
y = data_pd['label'].values

X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=0.70, stratify=y)
X_validate, X_test, y_validate, y_test = train_test_split(X_other, y_other, train_size=0.50, stratify=y_other)

print(
  'Train:\t\t{0}\nValidate:\t{1}\nTest:\t\t{2}'.format(
     len(y_train), 
     len(y_validate), 
     len(y_test)
    )
  )

# COMMAND ----------

# MAGIC %md ## Step 2: Tune Model Parameters
# MAGIC 
# MAGIC We've approached the task of identifying matches as a binary classification problem. There are many, many algorithms we might employ for such this kind of problem.  We have decided to use the [XGBoostClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier) given its demonstrated performance.
# MAGIC 
# MAGIC The downside of using XGBoost is that it's models employ a large number of hyperparameters, many of which interact to have sizeable impacts on model performance.  To overcome this problem, we will run through a number of iterations of the model using different hyperparameter settings.  To ensure this is efficient, we will make use of [hyperopt](http://hyperopt.github.io/hyperopt/) to intelligently search the hyperparameter values and [distribute](http://hyperopt.github.io/hyperopt/scaleout/spark/) the iterative work across the workers of our cluster:

# COMMAND ----------

# DBTITLE 1,Define Search Space for Hyperparameters
# define positive class scaling factor
weights = compute_class_weight(
  'balanced', 
  classes=np.unique(y_train), 
  y=y_train
  )
scale = weights[1]/weights[0]

# define hyperopt search space
search_space = {
    'max_depth' : hp.quniform('max_depth', 1, 30, 1)                                  # depth of trees (preference is for shallow trees or even stumps (max_depth=1))
    ,'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.40))     # learning rate for XGBoost
    ,'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001)                                   # minimum loss reduction required to make a further partition on a leaf node
    ,'min_child_weight' : hp.quniform('min_child_weight', 1, 20, 1)                   # minimum number of instances per node
    ,'subsample' : hp.loguniform('subsample', np.log(0.1), np.log(1.0))               # random selection of rows for training,
    ,'colsample_bytree' : hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)) # proportion of columns to use per tree
    ,'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0))# proportion of columns to use per level
    ,'colsample_bynode' : hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)) # proportion of columns to use per node
    ,'scale_pos_weight' : hp.loguniform('scale_pos_weight', np.log(1), np.log(scale * 10))   # weight to assign positive label to manage imbalance
    }

# define function to clean up parameter values generated by hyperopt
def clean_params(hyperopt_params):
  
  # configure model parameters
  params = hyperopt_params
  
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  
  # all other hyperparameters are taken as given by hyperopt
  
  # fixed parameters
  params['tree_method']='hist'
  params['predictor']='cpu_predictor'
  
  return params

# COMMAND ----------

# MAGIC %md Our hyperparameter settings will be sent to a function which will train a model using these settings and return a score.  Hyperopt will seek to minimize this score.  Because we are using the [average precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) to evaluate our model which produces a value between 0 and 1 and which improves as the score increases, we are multiplying this score by -1 to force hyperopt to maximize it:
# MAGIC 
# MAGIC **NOTE** For a broader discussion of average precision score in the context of class imbalances, please refer to the discussion in our notebook on [customer churn](https://databricks.com/notebooks/churn/3-model-selection.html).

# COMMAND ----------

# DBTITLE 1,Define Functions for Model Tuning
def evaluate_model(hyperopt_params):
  
  # clean params  
  params = clean_params(hyperopt_params)
  
  # instantiate model with parameters
  model = XGBClassifier(**params)
  
  # train
  model.fit(X_train_broadcast.value, y_train_broadcast.value)
  
  # predict
  y_prob = model.predict_proba(X_validate_broadcast.value)
  
  # score
  model_ap = average_precision_score(y_validate_broadcast.value, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)  # record actual metric with mlflow run
  
  # invert metric for hyperopt
  loss = -1 * model_ap  
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md With our search space and evaluation functions defined, hyperopt can now iteratively work towards an optimal solution:

# COMMAND ----------

# DBTITLE 1,Broadcast Sets for Model Training & Evaluation
# broadcast sets to workers for efficient model training
X_train_broadcast = sc.broadcast(X_train)
X_validate_broadcast = sc.broadcast(X_validate)

y_train_broadcast = sc.broadcast(y_train)
y_validate_broadcast = sc.broadcast(y_validate)

# COMMAND ----------

# DBTITLE 1,Tune Model
# perform evaluation
with mlflow.start_run(run_name='XGBClassifier'):
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=1000,
    trials=SparkTrials(parallelism=sc.defaultParallelism),
    verbose=True
    )

# COMMAND ----------

# MAGIC %md The optimal hyperparameter settings based on 1000 iterations were determined to be:

# COMMAND ----------

# DBTITLE 1,Review HyperParameters
hyperopt_params = space_eval(search_space, argmin)
hyperopt_params

# COMMAND ----------

# MAGIC %md We will now train our model using the hyperparameter settings identified in the steps above.  As we do so, we will persist the model to mlflow to make later deployments easier.  Because mlflow will wrap our model in such a way that we will only have access to its predict() method, we will override this method with a custom wrapper that allows predict() to return the probability of a positive class label which we believe to be more useful than a simple 0 or 1 response based on a fixed 0.50 probability threshold:

# COMMAND ----------

# DBTITLE 1,Train & Persist Final Model
model_name = 'abt_buy_product_match_model'

# shamelessly stolen from https://docs.databricks.com/_static/notebooks/mlflow/mlflow-end-to-end-example-aws.html
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

with mlflow.start_run(run_name='abt-buy product match model') as run:
  
  # use tuned params
  params = clean_params(hyperopt_params)
  
  # train
  model = XGBClassifier(**params)
  model.fit(X_train, y_train)
  #mlflow.sklearn.log_model(model, 'model')  # persist model with mlflow
  
  wrapped_model = SklearnModelWrapper(model)
  
  mlflow.pyfunc.log_model(
    artifact_path='model', 
    python_model=wrapped_model
    )
  
  # register last deployed model with mlflow model registry
  mv = mlflow.register_model(
      'runs:/{0}/model'.format(run.info.run_id),
      model_name
      )
  model_version = mv.version

  # predict
  y_prob = model.predict_proba(X_test)

  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)

  print('Average Precision Score:\t{0:.5f}'.format(model_ap))

# COMMAND ----------

# MAGIC %md Unlike the pipelines and LSH models persisted earlier, our model is a true predictive model and as such can be managed through mlflow's [model registry](https://www.mlflow.org/docs/latest/model-registry.html). The registry allows us to move models through various stages as tests and other approval workflows are successfully executed. Because this notebook is an automated demonstration, we will simply push our model into production status through code:
# MAGIC 
# MAGIC **NOTE** The registry may need as much as 5 minutes to complete the backend work it takes to make a newly registered model available for production deployment.  Please allow a bit of time between the completion of the last cell and the execution of this next one for the following code to complete successfully.

# COMMAND ----------

# DBTITLE 1,Elevate Model to Production Status
# archive any production model versions (from any previous runs of this notebook or manual workflow management)
client = mlflow.tracking.MlflowClient()

for mv in client.search_model_versions("name='{0}'".format(model_name)):
    # if model with this name is marked production
    if mv.current_stage.lower() == 'production':
      # mark is as archived
      client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage='archived'
        )
      
# transition newly deployed model to production stage
client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage='production'
  )     

# COMMAND ----------

# MAGIC %md ## Step 3: Deploy Product Matching Pipeline
# MAGIC 
# MAGIC Now we need to consider deployment.  When we have new products to compare, we need to transform the name, description and pricing attributes into features and then quickly identify candidates for a match.  We have trained pipelines and LSH models to enable this which we can retrieve from mlflow.  Once we've produced our candidate pairs, we then can use our trained model, also retrievable from mlflow, to score the pairs as a potential match:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/entityres_workflow2.png' width=800>
# MAGIC 
# MAGIC To setup this workflow, let's start with the retrieval of our assets from mlflow:

# COMMAND ----------

# DBTITLE 1,Setup Dir to Hold Temp Assets
# mkdir to hold retrieved artifacts
retrieval_path = '/tmp/retrieval'
try:
  dbutils.fs.rm(retrieval_path, recurse=True)
except:
  pass
dbutils.fs.mkdirs(retrieval_path)

# COMMAND ----------

# DBTITLE 1,Retrieve Latest Pipelines
client = MlflowClient()
# get last pipeline run
last_pipeline_run = client.search_runs(
  experiment_ids=[experiment.experiment_id], 
  filter_string="tags.`mlflow.runName`='abt_buy_pipelines'"
  )[0]

# retrieve pipelines from mlflow
pipelines = {'name_pipeline':None, 'descript_pipeline':None, 'price_pipeline':None}
for pipeline in pipelines:
  
  for i in range(3): # error handling to deal with occasional timeouts
    try:
      # retrieve pipeline
      pipeline_path = client.download_artifacts(
          last_pipeline_run.info.run_id, 
          pipeline,
          dst_path='/dbfs'+retrieval_path
          )
    except:
      pass
      time.sleep(5)
    else:
      break

  pipelines[pipeline] = PipelineModel.load( pipeline_path[5:] ) # remove /dbfs from front of path

# present results
pipelines

# COMMAND ----------

# DBTITLE 1,Retrieve LSH Models
# get last pipeline run
last_lshmodel_run = client.search_runs(
  experiment_ids=[experiment.experiment_id], 
  filter_string="tags.`mlflow.runName`='abt_buy_lsh'"
  )[0]

# retrieve pipelines from mlflow
lshmodels = {'name_lsher':None, 'descript_lsher':None}
for lshmodel in lshmodels:
  
  for i in range(3): # error handling to deal with occasional timeouts
    try:
      # retrieve lsh models
      lshmodel_path = client.download_artifacts(
          last_lshmodel_run.info.run_id, 
          lshmodel, 
          dst_path='/dbfs'+retrieval_path
          )
    except:
      pass
      time.sleep(5)
    else:
      break
  

  lshmodels[lshmodel] = BucketedRandomProjectionLSHModel.load( lshmodel_path[5:] ) # remove /dbfs from front of path
  
lshmodels

# COMMAND ----------

# DBTITLE 1,Retrieve Classification Model
classification_model = mlflow.pyfunc.load_model(
    model_uri=f'models:/abt_buy_product_match_model/Production'
    )

# define function based on mlflow recorded model
match_score_udf = mlflow.pyfunc.spark_udf(
  spark, 
  'models:/abt_buy_product_match_model/Production', 
  result_type=DoubleType()
  )

# register the function for use in SQL
_ = spark.udf.register('match_score', match_score_udf)

# COMMAND ----------

# MAGIC %md With our assets retrieved, let's now setup a demonstration workflow.  We will pretend there are a few new products or newly modified products on the Abt-side of things by making a random selection of existing products:

# COMMAND ----------

# DBTITLE 1,Retrieve "New & Updated" Products
new_modified_products = spark.table('DELTA.`/tmp/matching/silver/abt`').sample(fraction=0.01).cache()
display(new_modified_products)

# COMMAND ----------

# MAGIC %md We might generate features for these products as follows: 

# COMMAND ----------

# DBTITLE 1,Generate Features
# structure data as required for pipeline
incoming_products = (
  new_modified_products
    .selectExpr("'abt' as table", 'id', 'name', 'description', 'price')
  )

# generate features for incoming products
incoming_features = (
  pipelines['price_pipeline'].transform(
    pipelines['descript_pipeline'].transform(
      pipelines['name_pipeline'].transform(incoming_products)
      )
    )
  ).select('table','id','name_features','descript_features','price_features')

# merge features with existing features dataset
target = DeltaTable.forPath(spark, '/tmp/matching/silver/abtbuyfeatures')
(
  target.alias('target')
    .merge(
      incoming_features.alias('source'),
      'source.table=target.table AND source.id=target.id'
      ) 
      .whenMatchedUpdate(set = {
        'name_features':'source.name_features', 
        'descript_features':'source.descript_features', 
        'price_features':'source.price_features'
        })
      .whenNotMatchedInsertAll()
      .execute()
  )

# COMMAND ----------

# MAGIC %md And now we identify the candidate pairs for these incoming products:

# COMMAND ----------

# DBTITLE 1,Identify Candidate Pairs
max_distance_from_target = 1.1

name_pairs = (
  lshmodels['name_lsher'].approxSimilarityJoin(
    lshmodels['name_lsher'].transform(incoming_features),
    lshmodels['name_lsher'].transform(
      spark.table('DELTA.`/tmp/matching/silver/abtbuyfeatures`').filter(f.expr("table='buy'"))
      ),
    threshold = max_distance_from_target,
    distCol = 'distance'
    ).selectExpr(
        'datasetA.table as tableA', 
        'datasetA.id as idA', 
        'datasetB.table as tableB', 
        'datasetB.id as idB', 
        'distance as name_distance'
        )
  )

descript_pairs = (
  lshmodels['descript_lsher'].approxSimilarityJoin(
    lshmodels['descript_lsher'].transform(incoming_features),
    lshmodels['descript_lsher'].transform(
      spark.table('DELTA.`/tmp/matching/silver/abtbuyfeatures`').filter(f.expr("table='buy'"))
      ),
    threshold = max_distance_from_target,
    distCol = 'distance'
    ).selectExpr(
        'datasetA.table as tableA', 
        'datasetA.id as idA', 
        'datasetB.table as tableB', 
        'datasetB.id as idB', 
        'distance as descript_distance'
        )
  )

pairs = (
  name_pairs.select('idA', 'idB')
    .union(descript_pairs.select('idA', 'idB'))
    .distinct()
  )

# COMMAND ----------

# MAGIC %md And now the distances associated with these pairs is calculated and a score is generated:

# COMMAND ----------

# DBTITLE 1,Define Function for Distance Calcs
# MAGIC %scala
# MAGIC  
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors}
# MAGIC  
# MAGIC // from https://github.com/karlhigley/spark-neighbors/blob/master/src/main/scala/com/github/karlhigley/spark/neighbors/linalg/DistanceMeasure.scala
# MAGIC val squared_dist = udf { (v1: Vector, v2: Vector) =>
# MAGIC  Vectors.sqdist(v1, v2)
# MAGIC }
# MAGIC 
# MAGIC spark.udf.register("squared_dist", squared_dist)

# COMMAND ----------

# DBTITLE 1,Calculate Scores
features = (
  pairs
    .join(spark.table('DELTA.`/tmp/matching/silver/abtbuyfeatures`').filter(f.expr("table='abt'")).alias('A'), on=f.expr('idA=A.id'))
    .join(spark.table('DELTA.`/tmp/matching/silver/abtbuyfeatures`').filter(f.expr("table='buy'")).alias('B'), on=f.expr('idB=B.id'))
    .withColumn('name_sqdist', f.expr('squared_dist(A.name_features, B.name_features)'))
    .withColumn('descript_sqdist', f.expr('squared_dist(A.descript_features, B.descript_features)'))
    .withColumn('price_sqdist', f.expr('squared_dist(A.price_features, B.price_features)'))
    .select('idA', 'idB', 'name_sqdist', 'descript_sqdist', 'price_sqdist')
    .withColumn('match_score', f.expr('match_score(name_sqdist, descript_sqdist, price_sqdist)'))
  )

display(
  features.orderBy(f.col('match_score').desc())
  )

# COMMAND ----------

# MAGIC %md The model does reasonably well but is not perfect.  There are multiple potential matches for some of the "new & updated" products and some straight misses.  Expert review and assessment will need to be provided on an on-going basis, but as more expert-assigned labels become available and additional attributes are considered, the reliability of the model should improve.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library / data source                  | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Abt-Buy                                | dataset                 | CC-BY 4.0  | https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution  |

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.conf import SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
import numpy as np
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark import SparkContext, RDD
# import related modules
def CreateSparkContext():
    def SetLogger( sc ):
        logger = sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
        logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
        logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

    sparkConf = SparkConf().setAppName("RunDecisionTreeBinary").set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf = sparkConf)
    print(("master="+sc.master))
    SetLogger(sc)
    return (sc)
sc = CreateSparkContext()
spark = SparkSession(sc)


pp_df = spark.read.csv("/usr/local/spark/data/hour.csv",header=True,inferSchema=True)
 #, 'dteday', 'yr', 'casual', 'registered').collect()
drop_list = ['instant', 'dteday', 'yr', 'casual', 'registered']
pp_df = pp_df.select([column for column in pp_df.columns if column not in drop_list])
# 11 relevant columns
data_rdd = pp_df.rdd.map(lambda x: LabeledPoint(x[10], x[:10])).collect()
# print(pp_df)
# data_array =  np.array(pp_df)
# data_array = pp_df.rdd.map(lambda x:x.stringFieldName.split(","))
distData = sc.parallelize(data_rdd)
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = distData.randomSplit([0.7, 0.3])

# pp_df.take(1)
# vectorAssembler=VectorAssembler(inputCols=["season","mnth","hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum"],outputCol="features")

model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    impurity='variance', maxDepth=10, maxBins=100)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)

rootMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() /\
    float(testData.count())
metrics = RegressionMetrics(labelsAndPredictions)
rmse = metrics.meanSquaredError
# print(predictions)
# print(rmse)
print(labelsAndPredictions)
print('Root Mean Squared Error = ' + str(rmse))
# print('Learned regression tree model:')
# print(model.toDebugString())

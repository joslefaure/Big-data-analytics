from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
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

pp_df.take(1)
vectorAssembler=VectorAssembler(inputCols=["season","mnth","hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum"],outputCol="features")

vpp_df = vectorAssembler.transform(pp_df)
splits = vpp_df.randomSplit([0.7,0.3])

train_df = splits[0]
train_df.count()
test_df = splits[1]
dt = DecisionTreeRegressor(featuresCol="features",labelCol="cnt", maxDepth = 10, maxBins = 100, impurity = 'variance')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction",metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
# df = pp_df.drop(pp_df.instant).collect()
# pp_df = pp_df.withColumn("predictions", lit(dt_predictions))
# pp_df.show(10)
dt_predictions.show(10)
print("RMSE: ", rmse)

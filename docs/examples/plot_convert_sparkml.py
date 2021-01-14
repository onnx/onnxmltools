# SPDX-License-Identifier: Apache-2.0

"""
.. _l-example-spark:

Converts a Spark model
======================

This example trains a `spark
<https://spark.apache.org/>`_
model on the Iris datasets and converts it
into ONNX.

.. contents::
    :local:

Train a model
+++++++++++++

"""
import os
import numpy
from pandas import DataFrame
import onnx
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import onnxruntime as rt
import skl2onnx
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.linalg import VectorUDT, SparseVector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_sparkml


def start_spark(options=None):
    import os
    import sys
    import pyspark
    executable = sys.executable
    os.environ["SPARK_HOME"] = pyspark.__path__[0]
    os.environ["PYSPARK_PYTHON"] = executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = executable

    builder = SparkSession.builder.appName("pyspark-unittesting").master("local[1]")
    if options:
        for k,v in options.items():
            builder.config(k, v)
    spark = builder.getOrCreate()

    return spark


def stop_spark(spark):
    spark.sparkContext.stop()

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
df = DataFrame(X_train, columns="x1 x2 x3 x4".split())
df['class'] = y_train
# df.to_csv("data_train.csv", index=False, header=False)

this_script_dir = os.path.abspath('.')
if os.name == 'nt' and os.environ.get('HADOOP_HOME') is None:
    print('setting HADOOP_HOME to: ', this_script_dir)
    os.environ['HADOOP_HOME'] = this_script_dir
spark_session = start_spark()


input_path = os.path.join(this_script_dir, "data_train.csv")

data = spark_session.createDataFrame(df)
feature_cols = data.columns[:-1]
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
train_data = assembler.transform(data)
train_data = train_data.select(['features', 'class'])
label_indexer = StringIndexer(inputCol='class', outputCol='label').fit(train_data)
train_data = label_indexer.transform(train_data)
train_data = train_data.select(['features', 'label'])
train_data.show(10)

lr = LogisticRegression(maxIter=100, tol=0.0001)
model = lr.fit(train_data)


###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

initial_types = [('features', FloatTensorType([None, 4]))]
onx = convert_sparkml(model, 'sparkml logistic regression', initial_types)

stop_spark(spark_session)

###################################
# Compute the predictions with onnxruntime
# ++++++++++++++++++++++++++++++++++++++++

sess = rt.InferenceSession(onx.SerializeToString())
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

##################################
# Display the ONNX graph
# ++++++++++++++++++++++
#
# Finally, let's see the graph converted with *onnxmltools*.
import os
import matplotlib.pyplot as plt
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

pydot_graph = GetPydotGraph(
    onx.graph, name=onx.graph.name, rankdir="TB",
    node_producer=GetOpNodeProducer(
        "docstring", color="yellow", fillcolor="yellow", style="filled"))
pydot_graph.write_dot("model.dot")

os.system('dot -O -Gdpi=300 -Tpng model.dot')

image = plt.imread("model.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis('off')


#################################
# **Versions used for this example**

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("onnxmltools: ", onnxmltools.__version__)
print("pyspark: ", pyspark.__version__)

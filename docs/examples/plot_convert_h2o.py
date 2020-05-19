# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-example-h2o:

Converts a H2O model
====================

This example trains a `h2o
<http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html>`_
model on the Iris datasets and converts it
into ONNX.

.. contents::
    :local:

Train a model
+++++++++++++

"""
import os
import numpy
import onnx
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import onnxruntime as rt
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import skl2onnx
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_h2o

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

h2o.init(port=54440)

f_train_x = h2o.H2OFrame(X_train)
xc = list(range(0, f_train_x.ncol))
yc = f_train_x.ncol
f_train_y = h2o.H2OFrame(y_train)
f_train = f_train_x.cbind(f_train_y.asfactor())

glm_logistic = H2OGradientBoostingEstimator(ntrees=10, max_depth=5)
glm_logistic.train(x=xc, y=yc, training_frame=f_train)

if not os.path.exists("model"):
    os.mkdir("model")
pth = glm_logistic.download_mojo(path="model")

###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_h2o(pth, initial_types=initial_type)

h2o.cluster().shutdown()

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
print("h2o: ", h2o.__version__)

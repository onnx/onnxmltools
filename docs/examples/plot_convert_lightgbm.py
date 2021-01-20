# SPDX-License-Identifier: Apache-2.0

"""
.. _l-example-lightgbm:

Converts a LightGBM model
=========================

This example trains a `LightGBM
<https://lightgbm.readthedocs.io/en/latest/>`_
model on the Iris datasets and converts it
into ONNX.

.. contents::
    :local:

Train a model
+++++++++++++

"""

import numpy
import onnx
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import lightgbm
from lightgbm import LGBMClassifier, Dataset, train as train_lgbm
import onnxruntime as rt
import skl2onnx
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LGBMClassifier()
clr.fit(X_train, y_train)
print(clr)

###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_lightgbm(clr, initial_types=initial_type)

###################################
# Compute the predictions with onnxruntime
# ++++++++++++++++++++++++++++++++++++++++

sess = rt.InferenceSession(onx.SerializeToString())
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

###############################################
# With Dataset
# ++++++++++++
#
# Huge datasets cannot be handled with the scikit-learn API.
# DMatrix must be used. Let's see how to convert the trained
# model.

dtrain = Dataset(X_train, label=y_train)

param = {'objective': 'multiclass', 'num_class': 3}
bst = train_lgbm(param, dtrain, 10)

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_lightgbm(bst, initial_types=initial_type)

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
print("lightgbm: ", lightgbm.__version__)

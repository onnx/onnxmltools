# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-example-libsvm:

Converts a libsvm model
=======================

This example trains a `libsvm
<https://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_
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
from svm import C_SVC as SVC
import svmutil
import onnxruntime as rt

import skl2onnx
import onnxmltools
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_libsvm

iris = load_iris()
X, y = iris.data, iris.target
y_multi = numpy.zeros((y.shape[0], 3), dtype=numpy.int64)
y_multi[:, y] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y_multi)

prob = svmutil.svm_problem(y, X.tolist())

param = svmutil.svm_parameter()
param.svm_type = SVC
param.kernel_type = svmutil.LINEAR
param.eps = 1
param.probability = 1

clr = svmutil.svm_train(prob, param)

###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_libsvm(clr, initial_types=initial_type)

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

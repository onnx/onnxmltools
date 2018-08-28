# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._registration import register_converter


def convert_sklearn_k_neighbors_regressor(scope, operator, container):
    knn = operator.raw_operator
    training_examples = knn._fit_X
    training_labels = knn._y
    distance_power = knn.p

    training_examples_name = scope.get_unique_variable_name('training_examples')
    labels_name = scope.get_unique_variable_name('training_labels')
    sub_results_name = scope.get_unique_variable_name('sub_results')
    distance_name = scope.get_unique_variable_name('distance')
    distance_power_name = scope.get_unique_variable_name('distance_power')
    reduced_sum_name = scope.get_unique_variable_name('reduced_sum')
    topk_values_name = scope.get_unique_variable_name('topk_values')
    topk_indices_name = scope.get_unique_variable_name('topk_indices')
    topk_labels_name = scope.get_unique_variable_name('topk_labels')

    container.add_initializer(training_examples_name, onnx_proto.TensorProto.FLOAT,
                              training_examples.shape, training_examples.flatten())
    container.add_initializer(distance_power_name, onnx_proto.TensorProto.FLOAT,
                              [], [distance_power])
    container.add_initializer(labels_name, onnx_proto.TensorProto.FLOAT,
                              training_labels.shape, training_labels)

    container.add_node('Sub', [operator.inputs[0].full_name, training_examples_name],
                       sub_results_name, name='Sub', op_version=7)
    container.add_node('Pow', [sub_results_name, distance_power_name],
                       distance_name, name='Pow')
    container.add_node('ReduceSum', distance_name,
                       reduced_sum_name, name='ReduceSum', axes=[1])
    container.add_node('TopK', reduced_sum_name,
                       [topk_values_name, topk_indices_name], name='TopK', k=knn.n_neighbors)
    container.add_node('ArrayFeatureExtractor', [labels_name, topk_indices_name],
                       topk_labels_name, name='ArrayFeatureExtractor', op_domain='ai.onnx.ml')
    container.add_node('ReduceMean', topk_labels_name, 
                       operator.output_full_names, name='ReduceMean')


register_converter('SklearnKNeighborsRegressor', convert_sklearn_k_neighbors_regressor)

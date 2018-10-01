# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._apply_operation import apply_abs, apply_mul, apply_reshape, apply_sub
from ...common._registration import register_converter
import numpy as np


def convert_sklearn_knn(scope, operator, container):
    # Computational graph:
    #
    # In the following graph, variable names are in lower case characters only
    # and operator names are in upper case characters. We borrow operator names 
    # from the official ONNX spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    # All variables are followed by their shape in [].
    #
    # Symbols:
    # M: Number of training set instances
    # N: Number of features
    # C: Number of classes
    # input: test set input
    # output: test set output 
    #
    # Graph:
    #
    #   input[1, N] ---> SUB <---- training_examples[M, N]
    #                     |
    #                     V
    #           sub_results[M, N] ----> POW <---- distance_power[1]
    #                                    |
    #                                    V
    # reduced_sum[M] <--- REDUCESUM <--- distance[M, N]
    #            |
    #            V
    # length -> RESHAPE -> reshaped_result[1, M]
    #                       |
    #                       V
    # n_neighbors[1] ----> TOPK
    #                       |    
    #                      / \
    #                     /   \
    #                     |    |
    #                     V    V
    #       topk_indices[K]   topk_values[K]
    #               |
    #               V
    #   ARRAYFEATUREEXTRACTOR <- training_labels[M]
    #           |
    #           V              (KNN Regressor)
    #          topk_labels[K] ------------------> REDUCEMEAN --> output[1] 
    #                    | 
    #                   /|\
    #                  / | \(KNN Classifier)
    #                 /  |  \
    #                /   |   \
    #               /    |    \__
    #               |    |       |
    #               V    V       V
    # label0 -> EQUAL  EQUAL ... EQUAL <- label(C-1)
    #            |       |          |
    #            V       V          V
    # output_label_0[C] ...       output_label_(C-1)[C]
    #            |       |          |
    #            V       V          V
    #          CAST    CAST    ... CAST 
    #            |       |          |
    #            V       V          V
    # output_cast_label_0[C] ...  output_cast_label_(C-1)[C]
    #            |       |          |
    #            V       V          V
    #      REDUCESUM  REDUCESUM ... REDUCESUM
    #            |       |          |
    #            V       V          V
    # output_label_reduced_0[1] ... output_label_reduced_(C-1)[1]
    #           \        |           /
    #            \____   |      ____/ 
    #                 \  |  ___/ 
    #                  \ | / 
    #                   \|/
    #                    V
    #                 CONCAT --> concat_labels[C]
    #                               |
    #                               V
    #                           ARGMAX --> predicted_label[1] 
    #                                       |
    #                                       V
    #            output[1] <--- ARRAYFEATUREEXTRACTOR <- classes[C]

    knn = operator.raw_operator
    training_examples = knn._fit_X
    training_labels = knn._y
    length = [1, len(training_labels)]
    distance_power = knn.p

    training_examples_name = scope.get_unique_variable_name('training_examples')
    training_labels_name = scope.get_unique_variable_name('training_labels')
    sub_results_name = scope.get_unique_variable_name('sub_results')
    abs_results_name = scope.get_unique_variable_name('abs_results')
    distance_name = scope.get_unique_variable_name('distance')
    distance_power_name = scope.get_unique_variable_name('distance_power')
    reduced_sum_name = scope.get_unique_variable_name('reduced_sum')
    topk_values_name = scope.get_unique_variable_name('topk_values')
    topk_indices_name = scope.get_unique_variable_name('topk_indices')
    topk_labels_name = scope.get_unique_variable_name('topk_labels')
    reshaped_result_name = scope.get_unique_variable_name('reshaped_result')
    negate_name = scope.get_unique_variable_name('negate')
    negated_reshaped_result_name = scope.get_unique_variable_name('negated_reshaped_result')
    
    container.add_initializer(training_examples_name, onnx_proto.TensorProto.FLOAT,
                              training_examples.shape, training_examples.flatten())
    container.add_initializer(distance_power_name, onnx_proto.TensorProto.FLOAT,
                              [], [distance_power])
    container.add_initializer(negate_name, onnx_proto.TensorProto.FLOAT,
                              [], [-1])

    if operator.type == 'SklearnKNeighborsRegressor':
        container.add_initializer(training_labels_name, onnx_proto.TensorProto.FLOAT,
                                  training_labels.shape, training_labels)

    apply_sub(scope, [operator.inputs[0].full_name, training_examples_name], sub_results_name, container, broadcast=1)
    apply_abs(scope, sub_results_name, abs_results_name, container)
    container.add_node('Pow', [abs_results_name, distance_power_name],
                       distance_name, name=scope.get_unique_operator_name('Pow'))
    container.add_node('ReduceSum', distance_name,
                       reduced_sum_name, name=scope.get_unique_operator_name('ReduceSum'), axes=[1])
    apply_reshape(scope, reduced_sum_name, reshaped_result_name, container, desired_shape=length)
    apply_mul(scope, [reshaped_result_name, negate_name], negated_reshaped_result_name, container, broadcast=1)
    container.add_node('TopK', negated_reshaped_result_name,
                       [topk_values_name, topk_indices_name], name=scope.get_unique_operator_name('TopK'), k=knn.n_neighbors)

    if operator.type == 'SklearnKNeighborsClassifier':
        raise NotImplementedError
    elif operator.type == 'SklearnKNeighborsRegressor':
        container.add_node('ArrayFeatureExtractor', [training_labels_name, topk_indices_name],
                           topk_labels_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor'), op_domain='ai.onnx.ml')
        container.add_node('ReduceMean', topk_labels_name, 
                           operator.output_full_names, name=scope.get_unique_operator_name('ReduceMean'))


register_converter('SklearnKNeighborsRegressor', convert_sklearn_knn)

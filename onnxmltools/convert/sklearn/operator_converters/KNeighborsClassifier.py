# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._registration import register_converter
import numpy as np


def convert_sklearn_k_neighbors_classifier(scope, operator, container):
    knn = operator.raw_operator
    training_examples = knn._fit_X
    training_labels = knn._y
    distance_power = knn.p
    classes = knn.classes_

    training_examples_name = scope.get_unique_variable_name('training_examples')
    training_labels_name = scope.get_unique_variable_name('training_labels')
    sub_results_name = scope.get_unique_variable_name('sub_results')
    distance_name = scope.get_unique_variable_name('distance')
    distance_power_name = scope.get_unique_variable_name('distance_power')
    reduced_sum_name = scope.get_unique_variable_name('reduced_sum')
    topk_values_name = scope.get_unique_variable_name('topk_values')
    topk_indices_name = scope.get_unique_variable_name('topk_indices')
    topk_labels_name = scope.get_unique_variable_name('topk_labels')
    concat_labels_name = scope.get_unique_variable_name('concat_labels')
    classes_name = scope.get_unique_variable_name('classes')
    predicted_label_name = scope.get_unique_variable_name('predicted_label')
    
    class_type = onnx_proto.TensorProto.STRING
    labels_name = [None] * len(classes)
    output_label_name = [None] * len(classes)
    output_cast_label_name = [None] * len(classes)
    output_label_reduced_name = [None] * len(classes)

    if np.issubdtype(knn.classes_.dtype, float):
        class_type = onnx_proto.TensorProto.FLOAT
    elif np.issubdtype(knn.classes_.dtype, int):
        class_type = onnx_proto.TensorProto.INT64

    for i in range(len(classes)):
        labels_name[i] = scope.get_unique_variable_name('class_labels_{}'.format(i))
        container.add_initializer(labels_name[i], class_type, 
                              [], [classes[i]])
        output_label_name[i] = scope.get_unique_variable_name('output_labels_{}'.format(i))
        output_cast_label_name[i] = scope.get_unique_variable_name('output_cast_labels_{}'.format(i))
        output_label_reduced_name[i] = scope.get_unique_variable_name('output_labels_reduced_{}'.format(i))

    container.add_initializer(training_examples_name, onnx_proto.TensorProto.FLOAT,
                              training_examples.shape, training_examples.flatten())
    container.add_initializer(distance_power_name, onnx_proto.TensorProto.FLOAT,
                              [], [distance_power])
    container.add_initializer(training_labels_name, class_type,
                              training_labels.shape, training_labels)
    container.add_initializer(classes_name, class_type, 
                              classes.shape, classes)

    container.add_node('Sub', [operator.inputs[0].full_name, training_examples_name],
                       sub_results_name, name='Sub')
    container.add_node('Pow', [sub_results_name, distance_power_name],
                       distance_name, name='Pow')
    container.add_node('ReduceSum', distance_name,
                       reduced_sum_name, name='ReduceSum', axes=[1])
    container.add_node('TopK', reduced_sum_name,
                       [topk_values_name, topk_indices_name], name='TopK', k=knn.n_neighbors)
    container.add_node('ArrayFeatureExtractor', [training_labels_name, topk_indices_name],
                       topk_labels_name, name='ArrayFeatureExtractor1', op_domain='ai.onnx.ml')
    for i in range(len(classes)):
        container.add_node('Equal', [labels_name[i], topk_labels_name],
                            output_label_name[i], op_version=7)
        container.add_node('Cast', output_label_name[i],
                            output_cast_label_name[i], to=onnx_proto.TensorProto.INT64, op_version=7)
        container.add_node('ReduceSum', output_cast_label_name[i],
                            output_label_reduced_name[i], axes=[1])

    container.add_node('Concat', [s for s in output_label_reduced_name],
                       concat_labels_name, name='Concat', axis=0, op_version=7)
    container.add_node('ArgMax', concat_labels_name, 
                       predicted_label_name, name='ArgMax')
    container.add_node('ArrayFeatureExtractor', [classes_name, predicted_label_name],
                       operator.outputs[0].full_name, name='ArrayFeatureExtractor2', op_domain='ai.onnx.ml')


register_converter('SklearnKNeighborsClassifier', convert_sklearn_k_neighbors_classifier)

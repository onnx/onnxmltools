# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
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
    # length -> RESHAPE -> reshaped[1, M]
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
    length_name = scope.get_unique_variable_name('length')
    reshaped_name = scope.get_unique_variable_name('reshaped')
    
    container.add_initializer(training_examples_name, onnx_proto.TensorProto.FLOAT,
                              training_examples.shape, training_examples.flatten())
    container.add_initializer(distance_power_name, onnx_proto.TensorProto.FLOAT,
                              [], [distance_power])
    container.add_initializer(length_name, onnx_proto.TensorProto.INT64,
                              [len(length)], length)

    if operator.type == 'SklearnKNeighborsRegressor':
        container.add_initializer(training_labels_name, onnx_proto.TensorProto.FLOAT,
                                  training_labels.shape, training_labels)

    container.add_node('Sub', [operator.inputs[0].full_name, training_examples_name],
                       sub_results_name, name=scope.get_unique_operator_name('Sub'), op_version=7)
    container.add_node('Abs', sub_results_name,
                       abs_results_name, name=scope.get_unique_operator_name('Abs'), op_version=6)
    container.add_node('Pow', [abs_results_name, distance_power_name],
                       distance_name, name=scope.get_unique_operator_name('Pow'))
    container.add_node('ReduceSum', distance_name,
                       reduced_sum_name, name=scope.get_unique_operator_name('ReduceSum'), axes=[1])
    container.add_node('Reshape', [reduced_sum_name, length_name],
                       reshaped_name, name=scope.get_unique_operator_name('Reshape'))
    container.add_node('TopK', reshaped_name,
                       [topk_values_name, topk_indices_name], name=scope.get_unique_operator_name('TopK'), k=knn.n_neighbors)

    if operator.type == 'SklearnKNeighborsClassifier':
        classes = knn.classes_
        concat_labels_name = scope.get_unique_variable_name('concat_labels')
        classes_name = scope.get_unique_variable_name('classes')
        predicted_label_name = scope.get_unique_variable_name('predicted_label')
        final_label_name = scope.get_unique_variable_name('final_label')
        
        class_type = onnx_proto.TensorProto.STRING
        labels_name = [None] * len(classes)
        output_label_name = [None] * len(classes)
        output_cast_label_name = [None] * len(classes)
        output_label_reduced_name = [None] * len(classes)

        if np.issubdtype(knn.classes_.dtype, float):
            class_type = onnx_proto.TensorProto.FLOAT
        elif np.issubdtype(knn.classes_.dtype, int):
            class_type = onnx_proto.TensorProto.INT32
        else:
            classes = np.array([s.encode('utf-8') for s in classes])

        for i in range(len(classes)):
            labels_name[i] = scope.get_unique_variable_name('class_labels_{}'.format(i))
            container.add_initializer(labels_name[i], onnx_proto.TensorProto.INT32,
                                  [], [i])
            output_label_name[i] = scope.get_unique_variable_name('output_label_{}'.format(i))
            output_cast_label_name[i] = scope.get_unique_variable_name('output_cast_label_{}'.format(i))
            output_label_reduced_name[i] = scope.get_unique_variable_name('output_label_reduced_{}'.format(i))

        container.add_initializer(classes_name, class_type, 
                                  classes.shape, classes)
        container.add_initializer(training_labels_name, onnx_proto.TensorProto.INT32,
                                  training_labels.shape, training_labels)

        container.add_node('ArrayFeatureExtractor', [training_labels_name, topk_indices_name],
                           topk_labels_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor1'), op_domain='ai.onnx.ml')
        for i in range(len(classes)):
            container.add_node('Equal', [labels_name[i], topk_labels_name],
                                output_label_name[i], op_version=7)
            # Casting to Int32 instead of Int64 as ReduceSum doesn't seem to support Int64 
            container.add_node('Cast', output_label_name[i],
                                output_cast_label_name[i], to=onnx_proto.TensorProto.INT32, op_version=7)
            container.add_node('ReduceSum', output_cast_label_name[i],
                                output_label_reduced_name[i], axes=[1])

        container.add_node('Concat', [s for s in output_label_reduced_name],
                           concat_labels_name, name=scope.get_unique_operator_name('Concat'), axis=0, op_version=7)
        container.add_node('ArgMax', concat_labels_name, 
                           predicted_label_name, name=scope.get_unique_operator_name('ArgMax'))
        if class_type == onnx_proto.TensorProto.INT32:
            container.add_node('ArrayFeatureExtractor', [classes_name, predicted_label_name],
                               final_label_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor2'), op_domain='ai.onnx.ml')
            container.add_node('Cast', final_label_name,
                                operator.outputs[0].full_name, to=onnx_proto.TensorProto.INT64, op_version=7)
        else:
            container.add_node('ArrayFeatureExtractor', [classes_name, predicted_label_name],
                               operator.outputs[0].full_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor2'), op_domain='ai.onnx.ml')
    elif operator.type == 'SklearnKNeighborsRegressor':
        container.add_node('ArrayFeatureExtractor', [training_labels_name, topk_indices_name],
                           topk_labels_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor'), op_domain='ai.onnx.ml')
        container.add_node('ReduceMean', topk_labels_name, 
                           operator.output_full_names, name=scope.get_unique_operator_name('ReduceMean'))


register_converter('SklearnKNeighborsClassifier', convert_sklearn_knn)
register_converter('SklearnKNeighborsRegressor', convert_sklearn_knn)

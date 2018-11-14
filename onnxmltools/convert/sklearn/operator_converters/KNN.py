# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._apply_operation import apply_abs, apply_cast, apply_mul, apply_reshape, apply_sub
from ...common._registration import register_converter
import numpy as np


def convert_sklearn_knn(scope, operator, container):
    # Computational graph:
    #
    # In the following graph, variable names are in lower case characters only
    # and operator names are in upper case characters. We borrow operator names 
    # from the official ONNX spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    # All variables are followed by their shape in [].
    # Note that KNN regressor and classifier share the same computation graphs until the top-k
    # nearest examples' labels (aka `topk_labels` in the graph below) are found.
    #
    # Symbols:
    # M: Number of training set instances
    # N: Number of features
    # C: Number of classes
    # input: input
    # output: output
    # output_prob (for KNN Classifier): class probabilities
    #
    # Graph:
    #
    #   input [1, N] --> SUB <---- training_examples [M, N]
    #                     |
    #                     V
    #           sub_results [M, N] ----> POW <---- distance_power [1]
    #                                     |
    #                                     V
    #  reduced_sum [M] <-- REDUCESUM <-- distance [M, N]
    #            |
    #            V
    # length -> RESHAPE -> reshaped_result [1, M]
    #                       |
    #                       V
    # n_neighbors [1] ----> TOPK
    #                       |    
    #                      / \
    #                     /   \
    #                     |    |
    #                     V    V
    #       topk_indices [K]   topk_values [K]
    #               |
    #               V
    #   ARRAYFEATUREEXTRACTOR <- training_labels [M]
    #           |
    #           V                        (KNN Regressor)
    #          topk_labels [K] ----------------------------> REDUCEMEAN --> output [1] 
    #                    |
    #                    |
    #                    | (KNN Classifier)                                      
    #                    |
    #                    |------------------------------------------------------.
    #                   /|\                    (probability calculation)        | 
    #                  / | \                                                    |
    #                 /  |  \ (label prediction)                                V 
    #                /   |   \                                                CAST
    #               /    |    \__                                               |
    #               |    |       |                                              V
    #               V    V       V                                   cast_pred_label [K, 1]
    # label0 -> EQUAL  EQUAL ... EQUAL <- label(C-1)                            |
    #            |       |          |                                           |
    #            V       V          V                                           |
    # output_label_0 [C] ...       output_label_(C-1) [C]                       |
    #            |       |          |                                           V
    #            V       V          V               pred_label_shape [2] --> RESHAPE    
    #          CAST    CAST    ... CAST                                         |
    #            |       |          |                                           V
    #            V       V          V                                reshaped_pred_label [K, 1]
    # output_cast_label_0 [C] ...  output_cast_label_(C-1) [C]                  |
    #            |       |          |                                           |
    #            V       V          V                                           |
    #      REDUCESUM  REDUCESUM ... REDUCESUM                                   |
    #            |       |          |                                           |
    #            V       V          V                                           |
    # output_label_reduced_0 [1] ... output_label_reduced_(C-1) [1]             |
    #           \        |           /                                          |
    #            \____   |      ____/                                           |
    #                 \  |  ___/                                                |
    #                  \ | /                                                    |
    #                   \|/                                                     |
    #                    V                                                      |
    #                 CONCAT --> concat_labels [C]                              |
    #                               |                                           |
    #                               V                                           |
    #                           ARGMAX --> predicted_label [1]                  |
    #                                       |                                   |
    #                                       V                                   |
    #            output [1] <--- ARRAYFEATUREEXTRACTOR <- classes [C]           |
    #                                                                           |
    #                                                                           |
    #                                                                           |
    #   ohe_model --> ONEHOTENCODER <-------------------------------------------'
    #                   |
    #                   V
    #  ohe_result [n_neighbors, C] -> REDUCEMEAN -> reduced_prob [1, C]
    #                                                |
    #                                                V
    #               output_probability [1, C]  <-  ZipMap 

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
        classes = knn.classes_
        concat_labels_name = scope.get_unique_variable_name('concat_labels')
        classes_name = scope.get_unique_variable_name('classes')
        predicted_label_name = scope.get_unique_variable_name('predicted_label')
        final_label_name = scope.get_unique_variable_name('final_label')
        reshaped_final_label_name = scope.get_unique_variable_name('reshaped_final_label')
        
        class_type = onnx_proto.TensorProto.STRING
        labels_name = [None] * len(classes)
        output_label_name = [None] * len(classes)
        output_cast_label_name = [None] * len(classes)
        output_label_reduced_name = [None] * len(classes)
        zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}

        if np.issubdtype(knn.classes_.dtype, np.floating):
            class_type = onnx_proto.TensorProto.INT32
            classes = np.array(list(map(lambda x: int(x), classes)))
            zipmap_attrs['classlabels_int64s'] = classes 
        elif np.issubdtype(knn.classes_.dtype, np.signedinteger):
            class_type = onnx_proto.TensorProto.INT32
            zipmap_attrs['classlabels_int64s'] = classes
        else:
            classes = np.array([s.encode('utf-8') for s in classes])    
            zipmap_attrs['classlabels_strings'] = classes

        for i in range(len(classes)):
            labels_name[i] = scope.get_unique_variable_name('class_labels_{}'.format(i))
            container.add_initializer(labels_name[i], onnx_proto.TensorProto.INT32, [], [i])
            output_label_name[i] = scope.get_unique_variable_name('output_label_{}'.format(i))
            output_cast_label_name[i] = scope.get_unique_variable_name('output_cast_label_{}'.format(i))
            output_label_reduced_name[i] = scope.get_unique_variable_name('output_label_reduced_{}'.format(i))

        container.add_initializer(classes_name, class_type, 
                                  classes.shape, classes)
        container.add_initializer(training_labels_name, onnx_proto.TensorProto.INT32,
                                  training_labels.shape, training_labels)

        container.add_node('ArrayFeatureExtractor', [training_labels_name, topk_indices_name], topk_labels_name,
                           name=scope.get_unique_operator_name('ArrayFeatureExtractor'), op_domain='ai.onnx.ml')
        for i in range(len(classes)):
            container.add_node('Equal', [labels_name[i], topk_labels_name],
                                output_label_name[i])
            # Casting to Int32 instead of Int64 as ReduceSum doesn't seem to support Int64 
            apply_cast(scope, output_label_name[i], output_cast_label_name[i], container,
                       to=onnx_proto.TensorProto.INT32)
            container.add_node('ReduceSum', output_cast_label_name[i],
                                output_label_reduced_name[i], axes=[1])

        container.add_node('Concat', [s for s in output_label_reduced_name],
                           concat_labels_name, name=scope.get_unique_operator_name('Concat'), axis=0)
        container.add_node('ArgMax', concat_labels_name, 
                           predicted_label_name, name=scope.get_unique_operator_name('ArgMax'))
        container.add_node('ArrayFeatureExtractor', [classes_name, predicted_label_name], final_label_name,
                           name=scope.get_unique_operator_name('ArrayFeatureExtractor'), op_domain='ai.onnx.ml')
        if class_type == onnx_proto.TensorProto.INT32:
            apply_reshape(scope, final_label_name, reshaped_final_label_name, container, desired_shape=[-1,])
            apply_cast(scope, reshaped_final_label_name, operator.outputs[0].full_name, container,
                       to=onnx_proto.TensorProto.INT64)
        else:
            apply_reshape(scope, final_label_name, operator.outputs[0].full_name, container, desired_shape=[-1,])

        # Calculation of class probability
        pred_label_shape = [-1]

        cast_pred_label_name = scope.get_unique_variable_name('cast_pred_label')
        reshaped_pred_label_name = scope.get_unique_variable_name('reshaped_pred_label')
        reduced_prob_name = scope.get_unique_variable_name('reduced_prob')
        ohe_result_name = scope.get_unique_variable_name('ohe_result')

        apply_cast(scope, topk_labels_name, cast_pred_label_name, container, to=onnx_proto.TensorProto.INT64)
        apply_reshape(scope, cast_pred_label_name, reshaped_pred_label_name, container, desired_shape=pred_label_shape)
        if class_type == onnx_proto.TensorProto.STRING:
            container.add_node('OneHotEncoder', reshaped_pred_label_name, ohe_result_name,
                     name=scope.get_unique_operator_name('OneHotEncoder'), cats_strings=classes,
                     op_domain='ai.onnx.ml')
        else:
            container.add_node('OneHotEncoder', reshaped_pred_label_name, ohe_result_name,
                     name=scope.get_unique_operator_name('OneHotEncoder'), cats_int64s=classes,
                     op_domain='ai.onnx.ml')

        container.add_node('ReduceMean', ohe_result_name, 
                           reduced_prob_name, name=scope.get_unique_operator_name('ReduceMean'), axes=[0])
        container.add_node('ZipMap', reduced_prob_name, operator.outputs[1].full_name,
                           op_domain='ai.onnx.ml', **zipmap_attrs)
    elif operator.type == 'SklearnKNeighborsRegressor':
        container.add_node('ArrayFeatureExtractor', [training_labels_name, topk_indices_name],
                           topk_labels_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                           op_domain='ai.onnx.ml')
        container.add_node('ReduceMean', topk_labels_name, 
                           operator.output_full_names, name=scope.get_unique_operator_name('ReduceMean'))


register_converter('SklearnKNeighborsClassifier', convert_sklearn_knn)
register_converter('SklearnKNeighborsRegressor', convert_sklearn_knn)

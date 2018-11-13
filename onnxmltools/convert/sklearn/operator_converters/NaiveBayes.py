# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._apply_operation import apply_add, apply_cast, apply_exp, apply_reshape, apply_sub
from ...common._registration import register_converter
import numpy as np


def convert_sklearn_naive_bayes(scope, operator, container):
    # Computational graph:
    #
    # Note: In the following graph, variable names are in lower case characters only
    # and operator names are in upper case characters. We borrow operator names 
    # from the official ONNX spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    # All variables are followed by their shape in [].
    #
    # Symbols:
    # M: Number of instances
    # N: Number of features
    # C: Number of classes
    # input(or x): input
    # output(or y): output (There are two paths for producing output, one for
    #               string labels and the other one for int labels) 
    # output_probability: class probabilties 
    # feature_log_prob: Empirical log probability of features given a class, P(x_i|y)
    # class_log_prior: Smoothed empirical log probability for each class
    #
    # Multinomial NB
    # Equation: 
    #   y = argmax (class_log_prior + X . feature_log_prob^T)
    #
    # Graph:
    #
    #   input [M, N] -> MATMUL <- feature_log_prob.T [N, C]
    #                    |
    #                    V
    #        matmul_result [M, C] -> CAST <- onnx_proto.TensorProto.FLOAT
    #                                |
    #                                V
    #                    cast_result [M, C] -> SUM <- class_log_prior [1, C]
    #                                          |
    #                                          V                                   
    #                            sum_result [M, C] -> ARGMAX -> argmax_output [M, 1] 
    #                                                            |                
    #                                                            V               
    #                              classes [C] -------> ARRAYFEATUREEXTRACTOR
    #                                                            |
    #                                                            V          (string labels)
    #                                  array_feature_extractor_result [M, 1] -----------------------------.
    #                                               (int labels) |                                        | 
    #                                                            V                                        |
    #      output_shape [1] -> RESHAPE <- cast2_result [M, 1] <- CAST(to=onnx_proto.TensorProto.FLOAT)    |
    #                            |                                                                        |
    #                            V                                                                        V
    #                       reshaped_result [M,]            .--------------------------------------- RESHAPE
    #                                   |                   |
    #                                   V                   V
    #  (to=onnx_proto.TensorProto.INT64)CAST --------> output [M,]
    #
    # Bernoulli NB
    # Equation:
    #   y = argmax (class_log_prior + \sum neg_prob + X . (feature_log_prob - neg_prob))
    #   neg_prob = log( 1 - e ^ feature_log_prob)
    #
    #   Graph:
    #
    #           .------------------------------------------------------------------------------. 
    #           |                                                                              |
    #  feature_log_prob.T [N, C] -> EXP -> exp_result [N, C]                                   |
    #                                      |                                                   |
    #                                      V                                                   V
    #                         constant -> SUB -> sub_result [N, C] -> LOG -> neg_prob [N, C] -> SUB
    #                                                                        |                 |
    #                                                                        V                 V 
    #  .---------------- sum_neg_prob [1, C] <------------------------ REDUCE_SUM        difference_matrix [N, C]
    #  |                                                                                       |
    #  |                     .-----------------------------------------------------------------' 
    #  |                     |
    #  |                     V
    #  |    input [M, N] -> MATMUL -> dot_product [M, C]
    #  |                                       |
    #  |                                       V
    #  '------------------------------------> SUM
    #                                          |
    #                                          V
    #  class_log_prior [1, C] -> SUM <- partial_sum_result [M, C]
    #                            |
    #                            V
    #                   sum_result [M, C] -> ARGMAX -> argmax_output [M, 1] 
    #                                                            |
    #                                                            V
    #                              classes [C] -------> ARRAYFEATUREEXTRACTOR
    #                                                            |
    #                                                            V          (string labels)
    #                                  array_feature_extractor_result [M, 1] --------------------------.
    #                                               (int labels) |                                     | 
    #                                                            V                                     |
    #      output_shape [1] -> RESHAPE <- cast2_result [M, 1] <- CAST(to=onnx_proto.TensorProto.FLOAT) | 
    #                          |                                                                       |
    #                          V                                                                       V
    #                       reshaped_result [M,]             .-------------------------------------RESHAPE
    #                                   |                    |
    #                                   V                    |
    # (to=onnx_proto.TensorProto.INT64)CAST -> output [M,] <-'
    #
    #
    # If model's binarize attribute is not null, then input of Bernoulli NB is produced by the following graph:
    #
    #    input [M, N] -> GREATER <- threshold [1]
    #       |              |
    #       |              V
    #       |       condition [M, N] -> CAST(to=onnx_proto.TensorProto.FLOAT) -> cast_values [M, N]
    #       |                                                                       |
    #       V                                                                       V
    #   CONSTANT_LIKE ---------------------------> zero_tensor [M, N] ------------> ADD
    #                                                                               |
    #                                                                               V
    #                                                   input [M, N] <- binarised_input [M, N]
    #
    # Sub-graph for probability calculation common to both Multinomial and Bernoulli Naive Bayes
    #
    #  sum_result [M, C] -> REDUCELOGSUMEXP -> reduce_log_sum_exp_resulti [M,] -.
    #         |                                                                  |
    #         |                                                                  V
    #         |                                         log_prob_shape [2] -> RESHAPE
    #         |                                                                  |
    #         '--------------> SUB <---- reshaped_log_prob [M, 1] <--------------'
    #                           |
    #                           V
    #                     log_prob [M, C] -> EXP -> prob_tensor [M, C] -.
    #                                                                   |
    #         output_probability [M, C] <- ZIPMAP <---------------------'

    nb = operator.raw_operator
    class_log_prior = nb.class_log_prior_.astype('float32').reshape((1, -1))
    feature_log_prob = nb.feature_log_prob_.T.astype('float32')
    classes = nb.classes_
    output_shape = [-1,]

    class_log_prior_name = scope.get_unique_variable_name('class_log_prior')
    feature_log_prob_name = scope.get_unique_variable_name('feature_log_prob')
    sum_result_name = scope.get_unique_variable_name('sum_result')
    cast_result_name = scope.get_unique_variable_name('cast_result')
    argmax_output_name = scope.get_unique_variable_name('argmax_output')
    cast2_result_name = scope.get_unique_variable_name('cast2_result')
    reshaped_result_name = scope.get_unique_variable_name('reshaped_result')
    classes_name = scope.get_unique_variable_name('classes')
    reduce_log_sum_exp_result_name = scope.get_unique_variable_name('reduce_log_sum_exp_result')
    log_prob_name = scope.get_unique_variable_name('log_prob')
    prob_tensor_name = scope.get_unique_variable_name('prob_tensor')
    array_feature_extractor_result_name = scope.get_unique_variable_name('array_feature_extractor_result')

    class_type = onnx_proto.TensorProto.STRING
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
    if np.issubdtype(nb.classes_.dtype, np.floating):
        class_type = onnx_proto.TensorProto.INT32
        classes = np.array(list(map(lambda x: int(x), classes)))
        zipmap_attrs['classlabels_int64s'] = classes 
    elif np.issubdtype(nb.classes_.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
        zipmap_attrs['classlabels_int64s'] = classes
    else:
        classes = np.array([s.encode('utf-8') for s in classes])
        zipmap_attrs['classlabels_strings'] = classes

    container.add_initializer(feature_log_prob_name, onnx_proto.TensorProto.FLOAT,
                              feature_log_prob.shape, feature_log_prob.flatten())
    container.add_initializer(classes_name, class_type, classes.shape, classes)
    container.add_initializer(class_log_prior_name, onnx_proto.TensorProto.FLOAT,
                              class_log_prior.shape, class_log_prior.flatten())

    if operator.type == 'SklearnMultinomialNB':
        matmul_result_name = scope.get_unique_variable_name('matmul_result')
        shape_result_name = scope.get_unique_variable_name('shape_result')
        reshape_result_name = scope.get_unique_variable_name('reshape_result')

        container.add_node('MatMul', [operator.inputs[0].full_name, feature_log_prob_name],
                           matmul_result_name, name=scope.get_unique_operator_name('MatMul'))
        # Cast is required here as Sum op doesn't work with Float64
        apply_cast(scope, matmul_result_name, cast_result_name, container,
                   to=onnx_proto.TensorProto.FLOAT)
        container.add_node('Shape', class_log_prior_name, shape_result_name)
        container.add_node('Reshape', [cast_result_name, shape_result_name], reshape_result_name)
        container.add_node('Sum', [reshape_result_name, class_log_prior_name],
                           sum_result_name, name=scope.get_unique_operator_name('Sum'))
    else:
        constant_name = scope.get_unique_variable_name('constant')
        exp_result_name = scope.get_unique_variable_name('exp_result')
        sub_result_name = scope.get_unique_variable_name('sub_result')
        neg_prob_name = scope.get_unique_variable_name('neg_prob')
        sum_neg_prob_name = scope.get_unique_variable_name('sum_neg_prob')
        difference_matrix_name = scope.get_unique_variable_name('difference_matrix')
        dot_prod_name = scope.get_unique_variable_name('dot_prod')
        partial_sum_result_name = scope.get_unique_variable_name('partial_sum_result')

        container.add_initializer(constant_name, onnx_proto.TensorProto.FLOAT,
                                  [], [1.0])
        input_name = operator.inputs[0].full_name

        if nb.binarize is not None:
            threshold_name = scope.get_unique_variable_name('threshold')
            condition_name = scope.get_unique_variable_name('condition')
            cast_values_name = scope.get_unique_variable_name('cast_values')
            zero_tensor_name = scope.get_unique_variable_name('zero_tensor')
            binarised_input_name = scope.get_unique_variable_name('binarised_input')

            container.add_initializer(threshold_name, onnx_proto.TensorProto.FLOAT,
                                      [1], [nb.binarize])
        
            container.add_node('Greater', [operator.inputs[0].full_name, threshold_name],
                              condition_name, name=scope.get_unique_operator_name('Greater'), op_version=9)
            apply_cast(scope, condition_name, cast_values_name, container,
                       to=onnx_proto.TensorProto.FLOAT)
            container.add_node('ConstantLike', operator.inputs[0].full_name, zero_tensor_name,
                               name=scope.get_unique_operator_name('ConstantLike'),
                               dtype=onnx_proto.TensorProto.FLOAT, op_version=9)
            apply_add(scope, [zero_tensor_name, cast_values_name], binarised_input_name, container, broadcast=1)
            input_name = binarised_input_name

        apply_exp(scope, feature_log_prob_name, exp_result_name, container)
        apply_sub(scope, [constant_name, exp_result_name], sub_result_name, container, broadcast=1)
        container.add_node('Log', sub_result_name,
                           neg_prob_name, name=scope.get_unique_operator_name('Log'))
        container.add_node('ReduceSum', neg_prob_name,
                           sum_neg_prob_name, name=scope.get_unique_operator_name('ReduceSum'), axes=[0])
        apply_sub(scope, [feature_log_prob_name, neg_prob_name], difference_matrix_name, container)
        container.add_node('MatMul', [input_name, difference_matrix_name],
                           dot_prod_name, name=scope.get_unique_operator_name('MatMul'))
        container.add_node('Sum', [sum_neg_prob_name, dot_prod_name],
                           partial_sum_result_name, name=scope.get_unique_operator_name('Sum'))
        container.add_node('Sum', [partial_sum_result_name, class_log_prior_name],
                           sum_result_name, name=scope.get_unique_operator_name('Sum'))

    container.add_node('ArgMax', sum_result_name,
                       argmax_output_name, name=scope.get_unique_operator_name('ArgMax'), axis=1)

    # Calculation of class probability
    log_prob_shape = [-1, 1]

    reshaped_log_prob_name = scope.get_unique_variable_name('reshaped_log_prob')

    container.add_node('ReduceLogSumExp', sum_result_name,
                       reduce_log_sum_exp_result_name, name=scope.get_unique_operator_name('ReduceLogSumExp'),
                       axes=[1], keepdims=0)
    apply_reshape(scope, reduce_log_sum_exp_result_name, reshaped_log_prob_name, container, desired_shape=log_prob_shape)
    apply_sub(scope, [sum_result_name, reshaped_log_prob_name], log_prob_name, container, broadcast=1)
    apply_exp(scope, log_prob_name, prob_tensor_name, container)
    container.add_node('ZipMap', prob_tensor_name, operator.outputs[1].full_name,
                           op_domain='ai.onnx.ml', **zipmap_attrs)

    container.add_node('ArrayFeatureExtractor', [classes_name, argmax_output_name],
                       array_feature_extractor_result_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor'), op_domain='ai.onnx.ml')
    # Reshape op does not seem to handle INT64 tensor even though it is listed as one of the
    # supported types in the doc, so Cast was required here.
    if class_type == onnx_proto.TensorProto.INT32: # int labels
        container.add_node('Cast', array_feature_extractor_result_name, 
                            cast2_result_name, to=onnx_proto.TensorProto.FLOAT, op_version=7)
        apply_reshape(scope, cast2_result_name, reshaped_result_name, container, desired_shape=output_shape)
        container.add_node('Cast', reshaped_result_name, 
                            operator.outputs[0].full_name, to=onnx_proto.TensorProto.INT64, op_version=7)
    else: # string labels
        apply_reshape(scope, array_feature_extractor_result_name, operator.outputs[0].full_name, container,
                      desired_shape=output_shape)

register_converter('SklearnMultinomialNB', convert_sklearn_naive_bayes)
register_converter('SklearnBernoulliNB', convert_sklearn_naive_bayes)

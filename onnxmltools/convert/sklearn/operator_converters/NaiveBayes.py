# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._registration import register_converter


def convert_sklearn_naive_bayes(scope, operator, container):
    # Computational graph:
    #
    # Note: In the following graph, variable names are in lower case characters only
    # and operator names are in upper case characters. We borrow operator names 
    # from the official ONNX spec: https://github.com/onnx/onnx/blob/master/docs/Operators.md
    # All variables are followed by their shape in [].
    #
    # Symbols:
    # M: Number of test set instances
    # N: Number of features
    # C: Number of classes
    # input(or x): test set input
    # output(or y): test set output 
    # feature_log_prob: Empirical log probability of features given a class, P(x_i|y)
    # class_log_prior: Smoothed empirical log probability for each class
    #
    # Multinomial NB
    # Equation: 
    #   y = argmax (class_log_prior + X . feature_log_prob^T)
    #
    # Graph:
    #
    #   input[M, N] -> MATMUL <- feature_log_prob.T[N, C]
    #                    |
    #                    V
    #        matmul_result[M, C] -> CAST <- onnx_proto.TensorProto.FLOAT
    #                                |
    #                                V
    #                    cast_result[M, C] -> SUM <- class_log_prior[1, C]
    #                                          |
    #                                          V
    #                             sum_result[M, C] -> ARGMAX -> argmax_output[M, 1]
    #                                                            |
    #                                                            V
    #      output_shape[1] -> RESHAPE <- cast2_result[M, 1] <- CAST <- onnx_proto.TensorProto.FLOAT
    #                          |
    #                          V
    #                       reshaped_result[M,]
    #                          |
    #                          V
    #           output[M,] <- CAST <- onnx_proto.TensorProto.INT64
    #
    # Bernoulli NB
    # Equation:
    #   y = argmax (class_log_prior + Σ neg_prob - X . neg_prob)
    #   neg_prob = log( 1 - e ^ feature_log_prob)
    #
    #   Graph:
    #
    #  feature_log_prob.T[N, C] -> EXP -> exp_result[N, C] 
    #                                      |
    #                                      V
    #                         constant -> SUB -> sub_result[N, C] -> LOG -> neg_prob[N, C]
    #                                                                        |
    #                                                                        V
    #  ----------------- sum_neg_prob[1, C] <------------------------ REDUCE_SUM
    #  |                     |
    #  |                     V
    #  |    input[M, N] -> MATMUL -> inp_neg_prob_prod[M, C] -> CAST <- onnx_proto.TensorProto.FLOAT
    #  |                                                         |
    #  |                                                         V
    #  --------------------------------------> SUB  <- cast_result[M, C]
    #                                           |
    #                                           V
    #  class_log_prior[1, C] -> SUM <- difference_matrix[M, C]
    #                            |
    #                            V
    #                       sum_result[M, C] -> ARGMAX -> argmax_output[M, 1] 
    #                                                            |
    #                                                            V
    #      output_shape[1] -> RESHAPE <- cast2_result[M, 1] <- CAST <- onnx_proto.TensorProto.FLOAT
    #                          |
    #                          V
    #                       reshaped_result[M,]
    #                          |
    #                          V
    #           output[M,] <- CAST <- onnx_proto.TensorProto.INT64

    nb = operator.raw_operator
    class_log_prior = nb.class_log_prior_.astype('float32').reshape((1, -1))
    feature_log_prob = nb.feature_log_prob_.T.astype('float32')
    output_shape = [-1,]

    class_log_prior_name = scope.get_unique_variable_name('class_log_prior')
    feature_log_prob_name = scope.get_unique_variable_name('feature_log_prob')
    sum_result_name = scope.get_unique_variable_name('sum_result')
    cast_result_name = scope.get_unique_variable_name('cast_result')
    output_shape_name = scope.get_unique_variable_name('output_shape')
    argmax_output_name = scope.get_unique_variable_name('argmax_output')
    cast2_result_name = scope.get_unique_variable_name('cast2_result')
    reshaped_result_name = scope.get_unique_variable_name('reshaped_result')

    container.add_initializer(output_shape_name, onnx_proto.TensorProto.INT64,
                              [len(output_shape)], output_shape)
    container.add_initializer(class_log_prior_name, onnx_proto.TensorProto.FLOAT,
                              class_log_prior.shape, class_log_prior.flatten())
    container.add_initializer(feature_log_prob_name, onnx_proto.TensorProto.FLOAT,
                              feature_log_prob.shape, feature_log_prob.flatten())

    if operator.type == 'SklearnMultinomialNB':
        matmul_result_name = scope.get_unique_variable_name('matmul_result')

        container.add_node('MatMul', [operator.inputs[0].full_name, feature_log_prob_name],
                           matmul_result_name, name=scope.get_unique_operator_name('MatMul'))
        # Cast is required here as Sum op doesn't work with Float64
        container.add_node('Cast', matmul_result_name, 
                            cast_result_name, to=onnx_proto.TensorProto.FLOAT, op_version=7)
        container.add_node('Sum', [cast_result_name, class_log_prior_name],
                           sum_result_name, name=scope.get_unique_operator_name('Sum'))
    else:
        constant_name = scope.get_unique_variable_name('constant')
        exp_result_name = scope.get_unique_variable_name('exp_result')
        sub_result_name = scope.get_unique_variable_name('sub_result')
        neg_prob_name = scope.get_unique_variable_name('neg_prob')
        sum_neg_prob_name = scope.get_unique_variable_name('sum_neg_prob')
        inp_neg_prob_prod_name = scope.get_unique_variable_name('inp_neg_prob_prod')
        difference_matrix_name = scope.get_unique_variable_name('difference_matrix')

        container.add_initializer(constant_name, onnx_proto.TensorProto.FLOAT,
                                  [], [1.0])

        container.add_node('Exp', feature_log_prob_name,
                           exp_result_name, name=scope.get_unique_operator_name('Exp'), op_version=6)
        container.add_node('Sub', [constant_name, exp_result_name],
                           sub_result_name, name=scope.get_unique_operator_name('Sub1'))
        container.add_node('Log', sub_result_name,
                           neg_prob_name, name=scope.get_unique_operator_name('Log'))
        container.add_node('ReduceSum', neg_prob_name,
                           sum_neg_prob_name, name=scope.get_unique_operator_name('ReduceSum'), axes=[0])
        container.add_node('MatMul', [operator.inputs[0].full_name, neg_prob_name],
                           inp_neg_prob_prod_name, name=scope.get_unique_operator_name('MatMul'))
        # Cast is required here as Sub op doesn't work with Float64
        container.add_node('Cast', inp_neg_prob_prod_name, 
                            cast_result_name, to=onnx_proto.TensorProto.FLOAT, op_version=7)
        container.add_node('Sub', [sum_neg_prob_name, cast_result_name],
                           difference_matrix_name, name=scope.get_unique_operator_name('Sub2'))
        container.add_node('Sum', [difference_matrix_name, class_log_prior_name],
                           sum_result_name, name=scope.get_unique_operator_name('Sum'))

    container.add_node('ArgMax', sum_result_name,
                       argmax_output_name, name=scope.get_unique_operator_name('ArgMax'), axis=1)
    # Reshape op does not seem to handle INT64 tensor even though it is listed as one of the
    # supported types in the doc, so Cast was required here.
    container.add_node('Cast', argmax_output_name, 
                        cast2_result_name, to=onnx_proto.TensorProto.FLOAT, op_version=7)
    container.add_node('Reshape', [cast2_result_name, output_shape_name], 
                        reshaped_result_name, name=scope.get_unique_operator_name('Reshape2'))
    container.add_node('Cast', reshaped_result_name, 
                        operator.outputs[0].full_name, to=onnx_proto.TensorProto.INT64, op_version=7)


register_converter('SklearnMultinomialNB', convert_sklearn_naive_bayes)
register_converter('SklearnBernoulliNB', convert_sklearn_naive_bayes)

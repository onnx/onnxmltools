# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._registration import register_converter
import numpy as np


def convert_sklearn_naive_bayes(scope, operator, container):
    nb = operator.raw_operator
    class_log_prior = nb.class_log_prior_.astype('float32')
    feature_log_prob = nb.feature_log_prob_.T.astype('float32')
    shape = [1, -1]
    output_shape = [-1,]

    class_log_prior_name = scope.get_unique_variable_name('class_log_prior')
    feature_log_prob_name = scope.get_unique_variable_name('feature_log_prob')
    sum_result_name = scope.get_unique_variable_name('sum_result')
    cast_result_name = scope.get_unique_variable_name('cast_result')
    reshaped_result_name = scope.get_unique_variable_name('reshaped_result')
    shape_name = scope.get_unique_variable_name('shape')
    output_shape_name = scope.get_unique_variable_name('output_shape')
    argmax_output_name = scope.get_unique_variable_name('argmax_output')
    cast2_result_name = scope.get_unique_variable_name('cast_result2')
    reshaped2_result_name = scope.get_unique_variable_name('reshaped2_result')

    container.add_initializer(shape_name, onnx_proto.TensorProto.INT64,
                              [len(shape)], shape)
    container.add_initializer(output_shape_name, onnx_proto.TensorProto.INT64,
                              [len(output_shape)], output_shape)
    container.add_initializer(class_log_prior_name, onnx_proto.TensorProto.FLOAT,
                              class_log_prior.shape, class_log_prior)
    container.add_initializer(feature_log_prob_name, onnx_proto.TensorProto.FLOAT,
                              feature_log_prob.shape, feature_log_prob.flatten())

    container.add_node('Reshape', [class_log_prior_name, shape_name], 
                        reshaped_result_name, name='Reshape')

    if operator.type == 'SklearnMultinomialNB':
        matmul_result_name = scope.get_unique_variable_name('matmul_result')

        container.add_node('MatMul', [operator.inputs[0].full_name, feature_log_prob_name],
                           matmul_result_name, name='MatMul')
        container.add_node('Cast', matmul_result_name, 
                            cast_result_name, to=onnx_proto.TensorProto.FLOAT, op_version=7)
        container.add_node('Sum', [cast_result_name, reshaped_result_name],
                           sum_result_name, name='Sum')
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
                           exp_result_name, name='Exp', op_version=6)
        container.add_node('Sub', [constant_name, exp_result_name],
                           sub_result_name, name='Sub1')
        container.add_node('Log', sub_result_name,
                           neg_prob_name, name='Log')
        container.add_node('ReduceSum', neg_prob_name,
                           sum_neg_prob_name, name='ReduceSum', axes=[0])
        container.add_node('MatMul', [operator.inputs[0].full_name, neg_prob_name],
                           inp_neg_prob_prod_name, name='MatMul')
        container.add_node('Cast', inp_neg_prob_prod_name, 
                            cast_result_name, to=onnx_proto.TensorProto.FLOAT, op_version=7)
        container.add_node('Sub', [sum_neg_prob_name, cast_result_name],
                           difference_matrix_name, name='Sub2')
        container.add_node('Sum', [difference_matrix_name, reshaped_result_name],
                           sum_result_name, name='Sum')

    container.add_node('ArgMax', sum_result_name,
                       argmax_output_name, name='ArgMax', axis=1)
    # Reshape op does not seem to handle INT64 tensor even though it is listed as one of the
    # supported types in the doc, so Cast was required here.
    container.add_node('Cast', argmax_output_name, 
                        cast2_result_name, to=onnx_proto.TensorProto.FLOAT, op_version=7)
    container.add_node('Reshape', [cast2_result_name, output_shape_name], 
                        reshaped2_result_name, name='Reshape2')
    container.add_node('Cast', reshaped2_result_name, 
                        operator.outputs[0].full_name, to=onnx_proto.TensorProto.INT64, op_version=7)


register_converter('SklearnMultinomialNB', convert_sklearn_naive_bayes)
register_converter('SklearnBernoulliNB', convert_sklearn_naive_bayes)

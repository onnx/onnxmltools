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
    class_log_prior = nb.class_log_prior_
    feature_log_prob = nb.feature_log_prob_.T

    class_log_prior_name = scope.get_unique_variable_name('class_log_prior')
    feature_log_prob_name = scope.get_unique_variable_name('feature_log_prob')
    sum_result_name = scope.get_unique_variable_name('sum_result')

    container.add_initializer(class_log_prior_name, onnx_proto.TensorProto.FLOAT,
                              class_log_prior.shape, class_log_prior)
    container.add_initializer(feature_log_prob_name, onnx_proto.TensorProto.FLOAT,
                              feature_log_prob.shape, feature_log_prob.flatten())

    if operator.type == 'SklearnMultinomialNB':
        matmul_result_name = scope.get_unique_variable_name('matmul_result')

        container.add_node('MatMul', [operator.inputs[0].full_name, feature_log_prob_name],
                           matmul_result_name, name='MatMul')
        container.add_node('Sum', [matmul_result_name, class_log_prior_name],
                           sum_result_name, name='Sum', op_version=8)
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
        container.add_node('Sub', [sum_neg_prob_name, inp_neg_prob_prod_name],
                           difference_matrix_name, name='Sub2')
        container.add_node('Sum', [difference_matrix_name, class_log_prior_name],
                           sum_result_name, name='Sum', op_version=8)

    container.add_node('ArgMax', sum_result_name,
                       operator.outputs[0].full_name, name='ArgMax', axis=1)


register_converter('SklearnMultinomialNB', convert_sklearn_naive_bayes)
register_converter('SklearnBernoulliNB', convert_sklearn_naive_bayes)

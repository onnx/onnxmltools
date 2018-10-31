# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._apply_operation import apply_exp, apply_reshape, apply_sub, apply_max, apply_min
from ...common._registration import register_converter
from ...common._container import ModelComponentContainer
import onnx
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
    # M: Number of test set instances
    # N: Number of features
    # C: Number of classes
    # input(or x): test set input
    # output(or y): test set output (There are two paths for producing output, one for
    #               string labels and the other one for int labels) 
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
    #                              classes[C] -------> ARRAYFEATUREEXTRACTOR
    #                                                            |
    #                                                            V          (string labels)
    #                                  array_feature_extractor_result[M, 1] ----------------------------
    #                                               (int labels) |                                      | 
    #                                                            V                                      |
    #      output_shape[1] -> RESHAPE <- cast2_result[M, 1] <- CAST(to=onnx_proto.TensorProto.FLOAT)    |
    #                          |                                                                        |
    #                          V                                                                        V
    #                       reshaped_result[M,]             |------------------------------------- RESHAPE
    #                                   |                   |
    #                                   V                   V
    #  onnx_proto.TensorProto.INT64 -> CAST --------> output[M,]
    #
    # Bernoulli NB
    # Equation:
    #   y = argmax (class_log_prior + \sum neg_prob - X . neg_prob)
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
    #  |    input[M, N] -> MATMUL -> inp_neg_prob_prod[M, C] -> CAST(to=onnx_proto.TensorProto.FLOAT)
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
    #                              classes[C] -------> ARRAYFEATUREEXTRACTOR
    #                                                            |
    #                                                            V          (string labels)
    #                                  array_feature_extractor_result[M, 1] ----------------------------
    #                                               (int labels) |                                     | 
    #                                                            V                                     |
    #      output_shape[1] -> RESHAPE <- cast2_result[M, 1] <- CAST(to=onnx_proto.TensorProto.FLOAT)   | 
    #                          |                                                                       |
    #                          V                                                                       |
    #                       reshaped_result[M,]             --------------------------------------RESHAPE
    #                                   |                   |
    #                                   V                   |
    #  onnx_proto.TensorProto.INT64 -> CAST -> output[M,] <-|

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
        zipmap_attrs['classlabels_strings'] = classes
        classes = np.array([s.encode('utf-8') for s in classes])

    container.add_initializer(feature_log_prob_name, onnx_proto.TensorProto.FLOAT,
                              feature_log_prob.shape, feature_log_prob.flatten())
    container.add_initializer(classes_name, class_type, classes.shape, classes)

    if operator.type == 'SklearnMultinomialNB':
        container.add_initializer(class_log_prior_name, onnx_proto.TensorProto.FLOAT,
                                  class_log_prior.shape, class_log_prior.flatten())
        matmul_result_name = scope.get_unique_variable_name('matmul_result')

        container.add_node('MatMul', [operator.inputs[0].full_name, feature_log_prob_name],
                           matmul_result_name, name=scope.get_unique_operator_name('MatMul'))
        # Cast is required here as Sum op doesn't work with Float64
        container.add_node('Cast', matmul_result_name, cast_result_name,
                           to=onnx_proto.TensorProto.FLOAT, op_version=7)
                           
        shape_result_name = scope.get_unique_variable_name('shape_result')
        container.add_node('Shape', class_log_prior_name, shape_result_name)
        reshape_result_name = scope.get_unique_variable_name('reshape_result')
        container.add_node('Reshape', [cast_result_name, shape_result_name], reshape_result_name)
        
        container.add_node('Sum', [reshape_result_name, class_log_prior_name],
                           sum_result_name, name=scope.get_unique_operator_name('Sum'))
    else:
        container.add_initializer(class_log_prior_name, onnx_proto.TensorProto.FLOAT,
                                  class_log_prior.shape, class_log_prior.flatten())
        inp_neg_prob_prod_name = scope.get_unique_variable_name('inp_neg_prob_prod')
                
        # https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/naive_bayes.py#L948
        # neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        # jll += self.class_log_prior_ + neg_prob.sum(axis=1)
        
        # We store: (self.feature_log_prob_ - neg_prob).T
        #           self.class_log_prior_ + neg_prob.sum(axis=1)
        nb_neg_prob = np.log( - np.exp(nb.feature_log_prob_) + 1)
        nb_neg_prob_T = (nb.feature_log_prob_ - nb_neg_prob).T
        nb_log_prior = nb.class_log_prior_ + nb_neg_prob.sum(axis=1)
        
        nb_log_prior_name = scope.get_unique_variable_name('nb_log_prior')
        container.add_initializer(nb_log_prior_name, onnx_proto.TensorProto.FLOAT,
                                  nb_log_prior.shape, nb_log_prior.flatten())
        
        nb_neg_prob_T_name = scope.get_unique_variable_name('nb_neg_prob_T')
        container.add_initializer(nb_neg_prob_T_name, onnx_proto.TensorProto.FLOAT,
                                  nb_neg_prob_T.shape, nb_neg_prob_T.flatten())
                                  
        # binarization
        if nb.binarize is not None:
            # Sign(X - th) is replaced by [ max(x - threshold, 0) ] where [.] is ceil
            threshold = scope.get_unique_variable_name('threashold')
            container.add_initializer(threshold, onnx_proto.TensorProto.FLOAT, [], [nb.binarize])

            constant_0 = scope.get_unique_variable_name('constant_0')
            shape = (feature_log_prob.shape[0], )
            container.add_initializer(constant_0, onnx_proto.TensorProto.FLOAT, shape,
                                      np.zeros(shape).ravel())

            constant_1 = scope.get_unique_variable_name('constant_1')
            container.add_initializer(constant_1, onnx_proto.TensorProto.FLOAT, shape,
                                      np.ones(shape).ravel())

            input_threshold = scope.get_unique_variable_name('input_threshold')
            apply_sub(scope, [operator.inputs[0].full_name, threshold], input_threshold, container, broadcast=1)

            max_input_threshold = scope.get_unique_variable_name('max_input_threshold')
            apply_max(scope, [input_threshold, constant_0], max_input_threshold, container)

            ceiled = scope.get_unique_variable_name('ceiled')
            container.add_node('Ceil', max_input_threshold, ceiled, name=scope.get_unique_operator_name('Ceil'))
            
            binarized = scope.get_unique_variable_name('binarized')
            apply_min(scope, [ceiled, constant_1], binarized, container)

            input_name = binarized
        else:
            input_name = operator.inputs[0].full_name
        
        # safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T) --> inp_neg_prob_prod_name
        container.add_node('MatMul', [input_name, nb_neg_prob_T_name],
                           inp_neg_prob_prod_name, name=scope.get_unique_operator_name('MatMul'))

        # safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T) + nb_log_prior --> sum_result_name
        x_reshaped = scope.get_unique_variable_name('x_reshaped')
        nb_log_prior_reshaped = scope.get_unique_variable_name('nb_log_prior_reshaped')
        opname = scope.get_unique_operator_name('Sum')
        apply_reshape(scope, inp_neg_prob_prod_name, x_reshaped, container, desired_shape=class_log_prior.shape)
        apply_reshape(scope, nb_log_prior_name, nb_log_prior_reshaped, container, desired_shape=class_log_prior.shape)
        container.add_node('Sum', [x_reshaped, nb_log_prior_reshaped], sum_result_name, name=opname)        
        
    container.add_node('ArgMax', sum_result_name,
                       argmax_output_name, name=scope.get_unique_operator_name('ArgMax'), axis=1)

    # Following four statements are for predicting probabilities
    container.add_node('ReduceLogSumExp', sum_result_name, reduce_log_sum_exp_result_name,
                       name=scope.get_unique_operator_name('ReduceLogSumExp'), axes=[1], keepdims=0)
    apply_sub(scope, [sum_result_name, reduce_log_sum_exp_result_name], log_prob_name, container, broadcast=1)
    apply_exp(scope, log_prob_name, prob_tensor_name, container)
    
    # prob_tensor_name = binarized
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

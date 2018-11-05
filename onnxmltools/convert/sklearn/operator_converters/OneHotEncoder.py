# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import collections
import numpy
from ....proto import onnx_proto
from ...common._registration import register_converter


def convert_sklearn_one_hot_encoder(scope, operator, container):
    op = operator.raw_operator
    C = operator.inputs[0].type.shape[1]
    categorical_feature_indices = [i for i, mat in enumerate(op.categories_) if mat is not None and len(mat) > 0]

    # encoded_slot_sizes[i] is the number of output coordinates associated with the ith categorical feature
    categorical_values_per_feature = []
    
    categorical_values_per_feature = []
    for cat in op.categories_:
        if cat is None and len(cat) == 0:
            continue
        if cat.dtype in (numpy.float32, numpy.float64, numpy.int32, numpy.int64):
            categorical_values_per_feature.append(list(cat.astype(numpy.int64)))
        elif cat.dtype in (numpy.str, numpy.unicode, numpy.object):
            categorical_values_per_feature.append([str(_) for _ in cat])
        else:
            raise TypeError("Categories must be int or strings not {0}.".format(cat.dtype))

    # Variable names produced by one-hot encoders. Each of them is the encoding result of a categorical feature.
    final_variable_names = []
    final_variable_lengths = []
    for i, cats in zip(categorical_feature_indices, categorical_values_per_feature):
        # Put a feature index we want to encode to a tensor
        index_variable_name = scope.get_unique_variable_name('target_index')
        container.add_initializer(index_variable_name, onnx_proto.TensorProto.INT64, [1], [i])

        # Extract the categorical feature from the original input tensor
        extracted_feature_name = scope.get_unique_variable_name('extracted_feature_at_' + str(i))
        extractor_type = 'ArrayFeatureExtractor'
        extractor_attrs = {'name': scope.get_unique_operator_name(extractor_type)}
        container.add_node(extractor_type, [operator.inputs[0].full_name, index_variable_name],
                           extracted_feature_name, op_domain='ai.onnx.ml', **extractor_attrs)

        # Encode the extracted categorical feature as a one-hot vector
        encoder_type = 'OneHotEncoder'
        encoder_attrs = {'name': scope.get_unique_operator_name(encoder_type), 'cats_int64s': cats}
        encoded_feature_name = scope.get_unique_variable_name('encoded_feature_at_' + str(i))
        container.add_node(encoder_type, extracted_feature_name, encoded_feature_name, op_domain='ai.onnx.ml',
                           **encoder_attrs)

        # Collect features produce by one-hot encoders
        final_variable_names.append(encoded_feature_name)
        # For each categorical value, the length of its encoded result is the number of all possible categorical values
        final_variable_lengths.append(len(cats))

    # If there are some features which are not processed by one-hot encoding, we extract them directly from the original
    # input and combine them with the outputs of one-hot encoders.
    passed_indices = [i for i in range(C) if i not in categorical_feature_indices]
    if len(passed_indices) > 0:
        passed_feature_name = scope.get_unique_variable_name('passed_through_features')
        extractor_type = 'ArrayFeatureExtractor'
        extractor_attrs = {'name': scope.get_unique_operator_name(extractor_type)}
        passed_indices_name = scope.get_unique_variable_name('passed_feature_indices')
        container.add_initializer(passed_indices_name, onnx_proto.TensorProto.INT64, [1], [len(passed_indices)])
        container.add_node(extractor_type, [operator.inputs[0].full_name, passed_indices_name],
                           passed_feature_name, op_domain='ai.onnx.ml', **extractor_attrs)
        final_variable_names.append(passed_feature_name)
        final_variable_lengths.append(1)

    # Combine encoded features and passed features
    collector_type = 'FeatureVectorizer'
    collector_attrs = {'name': scope.get_unique_operator_name(collector_type)}
    collector_attrs['inputdimensions'] = final_variable_lengths
    container.add_node(collector_type, final_variable_names, operator.outputs[0].full_name,
                       op_domain='ai.onnx.ml', **collector_attrs)


register_converter('SklearnOneHotEncoder', convert_sklearn_one_hot_encoder)


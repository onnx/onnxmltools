# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter, register_shape_calculator
from ....proto import onnx_proto
from ...common.data_types import FloatTensorType


def convert_sparkml_one_hot_encoder(scope, operator, container):
    op = operator.raw_operator
    C = operator.inputs[0].type.shape[1]

    # encoded_slot_sizes[i] is the number of output coordinates associated with the ith categorical feature

    # Variable names produced by one-hot encoders. Each of them is the encoding result of a categorical feature.
    final_variable_names = []
    final_variable_lengths = []
    for i in range(0, len(op.categorySizes)):
        catSize = op.categorySizes[i]
        cats = range(0, catSize)
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
        final_variable_lengths.append(catSize)

    # Combine encoded features and passed features
    collector_type = 'FeatureVectorizer'
    collector_attrs = {'name': scope.get_unique_operator_name(collector_type)}
    collector_attrs['inputdimensions'] = final_variable_lengths
    container.add_node(collector_type, final_variable_names, operator.outputs[0].full_name,
                       op_domain='ai.onnx.ml', **collector_attrs)


register_converter('pyspark.ml.feature.OneHotEncoderModel', convert_sparkml_one_hot_encoder)


def calculate_sparkml_one_hot_encoder_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C] ---> [N, C']
        2. [N, 'None'] ---> [N, 'None']
    '''
    op = operator.raw_operator

    # encoded_slot_sizes[i] is the number of output coordinates associated with the ith categorical feature.
    encoded_slot_sizes = op.categorySizes

    N = operator.inputs[0].type.shape[0]
    # Calculate the output feature length by replacing the count of categorical
    # features with their encoded widths
    if operator.inputs[0].type.shape[1] != 'None':
        C = operator.inputs[0].type.shape[1] - 1 + sum(encoded_slot_sizes)
    else:
        C = 'None'

    operator.outputs[0].type = FloatTensorType([N, C])


register_shape_calculator('pyspark.ml.feature.OneHotEncoderModel', calculate_sparkml_one_hot_encoder_output_shapes)


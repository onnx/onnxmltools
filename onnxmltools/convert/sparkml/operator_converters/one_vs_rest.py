# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from onnx import onnx_pb as onnx_proto
from ...common._apply_operation import apply_concat, apply_argmax
from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from .linear_classifier import convert_sparkml_linear_classifier
from ..ops_names import get_sparkml_operator_name
from ...common._registration import register_converter, register_shape_calculator


def convert_one_vs_rest(scope, operator, container):
    op = operator.raw_operator
    classifier_output_names = []
    # initializer needed to extract the 2nd value of probability array
    index_tensor = scope.get_unique_variable_name('index_tensor')
    container.add_initializer(index_tensor, onnx_proto.TensorProto.INT64, [1], [1])
    # OneVsRest could have different classifiers
    # all must have at least Probability values for us to do the argmax
    for sub_model in op.models:
        classifier_op = scope.declare_local_operator(get_sparkml_operator_name(type(sub_model)), sub_model)
        classifier_op.raw_params = operator.raw_params
        classifier_op.inputs = operator.inputs
        classifier_prediction_output = scope.declare_local_variable('classifier_prediction', FloatTensorType())
        classifier_probability_output = scope.declare_local_variable('classifier_probability', FloatTensorType())
        classifier_op.outputs.append(classifier_prediction_output)
        classifier_op.outputs.append(classifier_probability_output)
        convert_sparkml_linear_classifier(scope, classifier_op, container)
        classifier_op.is_evaluated = True
        single_feature_tensor = scope.get_unique_variable_name('single_feature_tensor')
        container.add_node('ArrayFeatureExtractor', [classifier_probability_output.full_name, index_tensor],
                           single_feature_tensor,
                           op_domain='ai.onnx.ml',
                           name=scope.get_unique_operator_name('ArrayFeatureExtractor'))
        classifier_output_names.append(single_feature_tensor)

    concatenated_probabilities = scope.get_unique_variable_name('concatenated_predictions_tensor')
    apply_concat(scope, classifier_output_names, concatenated_probabilities, container, axis=1)
    # to get Prediction from probability
    apply_argmax(scope, concatenated_probabilities, operator.outputs[0].full_name, container,
                 axis=1, keepdims=1)


register_converter('pyspark.ml.classification.OneVsRestModel', convert_one_vs_rest)


def calculate_one_vs_rest_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = Int64TensorType(shape=[N])


register_shape_calculator('pyspark.ml.classification.OneVsRestModel',
                          calculate_one_vs_rest_output_shapes)

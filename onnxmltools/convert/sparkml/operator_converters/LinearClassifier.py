# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter

def convert_sparkml_logitic_regression(scope, operator, container):
    op = operator.raw_operator
    op_type = 'LinearClassifier'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    coefficients = op.coefficients.toArray().tolist()
    intercepts = [op.intercept]
    if op.numClasses == 2:
        coefficients = list(map(lambda x: -1 * x, coefficients)) + coefficients
        intercepts = list(map(lambda x: -1 * x, intercepts)) + intercepts

    attrs['coefficients'] = coefficients
    attrs['intercepts'] = intercepts
    attrs['multi_class'] = 1
    attrs['post_transform'] = 'LOGISTIC'
    attrs["classlabels_ints"] = range(0, op.numClasses)

    label_name = operator.outputs[0].full_name
    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')

    container.add_node(op_type, operator.inputs[0].full_name,
                       [label_name, probability_tensor_name],
                       op_domain='ai.onnx.ml', **attrs)

    # Make sure the probability sum is 1 over all classes
    normalizer_type = 'Normalizer'
    normalizer_attrs = {'name': scope.get_unique_operator_name(normalizer_type), 'norm': 'L1'}
    container.add_node(normalizer_type, probability_tensor_name, operator.outputs[1].full_name,
                       op_domain='ai.onnx.ml', **normalizer_attrs)


register_converter('pyspark.ml.classification.LogisticRegressionModel', convert_sparkml_logitic_regression)

from onnxmltools.convert.common.data_types import Int64TensorType, FloatTensorType
from ...common._registration import register_shape_calculator
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types

def calculate_logistic_regression_output_shapes(operator):
    '''
     This operator maps an input feature vector into a scalar label if the number of outputs is one. If two outputs
     appear in this operator's output list, we should further generate a map storing all classes' probabilities.

     Allowed input/output patterns are
         1. [N, C] ---> [N, 1], A sequence of map

     '''
    class_count = operator.raw_operator.numClasses
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=[1, class_count])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]

    operator.outputs[0].type = Int64TensorType(shape=[N])
    operator.outputs[1].type = FloatTensorType([N, class_count])


register_shape_calculator('pyspark.ml.classification.LogisticRegressionModel', calculate_logistic_regression_output_shapes)
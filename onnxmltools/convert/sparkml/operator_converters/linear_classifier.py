# SPDX-License-Identifier: Apache-2.0

from pyspark.ml.classification import LogisticRegressionModel, LinearSVCModel

from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def convert_sparkml_linear_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'LinearClassifier'
    attrs = {'name': scope.get_unique_operator_name(op_type)}

    if op.numClasses == 2:
        coefficients = op.coefficients.toArray().tolist()
        intercepts = [op.intercept]
        coefficients = list(map(lambda x: -1 * x, coefficients)) + coefficients
        intercepts = list(map(lambda x: -1 * x, intercepts)) + intercepts
    else:
        coefficients = op.coefficientMatrix.toArray().ravel().tolist()
        intercepts = op.interceptVector.toArray().ravel().tolist()

    if isinstance(op, LogisticRegressionModel):
        if op.numClasses > 2:
            coefficients = op.coefficientMatrix.toArray().ravel().tolist()
            intercepts = op.interceptVector.toArray().ravel().tolist()
        attrs['post_transform'] = 'LOGISTIC'
    elif isinstance(op, LinearSVCModel):
        attrs['post_transform'] = 'NONE'
    else:
        if op.numClasses >= 2:
            attrs['post_transform'] = 'SOFTMAX'
        else:
            attrs['post_transform'] = 'LOGISTIC'

    attrs['coefficients'] = coefficients
    attrs['intercepts'] = intercepts
    attrs['multi_class'] = 1 if op.numClasses >= 2 else 0
    attrs["classlabels_ints"] = list(range(0, op.numClasses))
    import pprint
    pprint.pprint(attrs)

    label_name = operator.outputs[0].full_name
    if not isinstance(operator.raw_operator, LinearSVCModel):
        probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
        container.add_node(op_type, operator.inputs[0].full_name,
                           [label_name, probability_tensor_name],
                           op_domain='ai.onnx.ml', **attrs)

        # Make sure the probability sum is 1 over all classes
        normalizer_type = 'Normalizer'
        normalizer_attrs = {'name': scope.get_unique_operator_name(normalizer_type), 'norm': 'L1'}
        container.add_node(normalizer_type, probability_tensor_name, operator.outputs[1].full_name,
                           op_domain='ai.onnx.ml', **normalizer_attrs)
    else:
        # add a dummy output variable since onnx LinearClassifier has 2
        unused_probabilities_output = scope.get_unique_variable_name('probabilities')
        container.add_node(op_type, operator.inputs[0].full_name,
                           [label_name,unused_probabilities_output],
                           op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.classification.LogisticRegressionModel', convert_sparkml_linear_classifier)
register_converter('pyspark.ml.classification.LinearSVCModel', convert_sparkml_linear_classifier)


def calculate_linear_classifier_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = Int64TensorType(shape=[N])
    if not isinstance(operator.raw_operator, LinearSVCModel):
        class_count = operator.raw_operator.numClasses
        operator.outputs[1].type = FloatTensorType([N, class_count])


register_shape_calculator('pyspark.ml.classification.LogisticRegressionModel', calculate_linear_classifier_output_shapes)
register_shape_calculator('pyspark.ml.classification.LinearSVCModel', calculate_linear_classifier_output_shapes)

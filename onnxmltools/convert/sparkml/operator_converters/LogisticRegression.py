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
    normalized_probability_tensor_name = scope.get_unique_variable_name(probability_tensor_name + '_normalized')
    normalizer_type = 'Normalizer'
    normalizer_attrs = {'name': scope.get_unique_operator_name(normalizer_type), 'norm': 'L1'}
    container.add_node(normalizer_type, probability_tensor_name, normalized_probability_tensor_name,
                       op_domain='ai.onnx.ml', **normalizer_attrs)

    # Post-process probability tensor produced by LinearClassifier operator
    zipmap_type = 'ZipMap'
    zipmap_attrs = {'name': scope.get_unique_operator_name(zipmap_type)}
    zipmap_attrs['classlabels_int64s'] = range(0, op.numClasses)

    container.add_node(zipmap_type, normalized_probability_tensor_name, operator.outputs[1].full_name,
                       op_domain='ai.onnx.ml', **zipmap_attrs)

register_converter('pyspark.ml.classification.LogisticRegressionModel', convert_sparkml_logitic_regression)

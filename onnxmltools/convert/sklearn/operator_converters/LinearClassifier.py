# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import six, numbers
from ...common._registration import register_converter


def convert_sklearn_linear_classifier(scope, operator, container):
    op = operator.raw_operator
    coefficients = op.coef_.flatten().tolist()
    intercepts = op.intercept_.tolist()
    classes = op.classes_
    if len(classes) == 2:
        coefficients = list(map(lambda x: -1 * x, coefficients)) + coefficients
        intercepts = list(map(lambda x: -1 * x, intercepts)) + intercepts

    multi_class = 0
    if hasattr(op, 'multi_class'):
        if op.multi_class == 'ovr':
            multi_class = 1
        else:
            multi_class = 2

    classifier_type = 'LinearClassifier'
    classifier_attrs = {'name': scope.get_unique_operator_name(classifier_type)}

    # nb = NodeBuilder(context, 'LinearClassifier', op_domain='ai.onnx.ml')
    classifier_attrs['coefficients'] = coefficients
    classifier_attrs['intercepts'] = intercepts
    classifier_attrs['multi_class'] = 1 if multi_class == 2 else 0
    if op.__class__.__name__ == 'LinearSVC':
        classifier_attrs['post_transform'] = 'NONE'
    else:
        if multi_class == 2:
            classifier_attrs['post_transform'] = 'SOFTMAX'
        else:
            classifier_attrs['post_transform'] = 'LOGISTIC'

    if all(isinstance(i, (six.string_types, six.text_type)) for i in classes):
        class_labels = [str(i) for i in classes]
        classifier_attrs['classlabels_strings'] = class_labels
    elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        classifier_attrs['classlabels_ints'] = class_labels
    else:
        raise RuntimeError('Label vector must be a string or a integer tensor')

    label_name = operator.outputs[0].full_name
    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')

    container.add_node(classifier_type, operator.inputs[0].full_name, [label_name, probability_tensor_name],
                       op_domain='ai.onnx.ml', **classifier_attrs)

    # Make sure the probability sum is 1 over all classes
    if multi_class > 0 and op.__class__.__name__ != 'LinearSVC':
        normalized_probability_tensor_name = scope.get_unique_variable_name(probability_tensor_name + '_normalized')
        normalizer_type = 'Normalizer'
        normalizer_attrs = {'name': scope.get_unique_operator_name(normalizer_type), 'norm': 'L1'}
        container.add_node(normalizer_type, probability_tensor_name, normalized_probability_tensor_name,
                           op_domain='ai.onnx.ml', **normalizer_attrs)
    else:
        normalized_probability_tensor_name = probability_tensor_name

    # Post-process probability tensor produced by LinearClassifier operator
    if len(class_labels) > 2 or op.__class__.__name__ != 'LinearSVC':
        zipmap_type = 'ZipMap'
        zipmap_attrs = {'name': scope.get_unique_operator_name(zipmap_type)}
        if all(isinstance(i, (six.string_types, six.text_type)) for i in class_labels):
            zipmap_attrs['classlabels_strings'] = class_labels
        elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in class_labels):
            zipmap_attrs['classlabels_int64s'] = class_labels
        else:
            raise RuntimeError('Label vector must be a string or a integer tensor')

        container.add_node(zipmap_type, normalized_probability_tensor_name, operator.outputs[1].full_name,
                           op_domain='ai.onnx.ml', **zipmap_attrs)
    else:
        normalized_probability_tensor_name = probability_tensor_name
        score_selector_type = 'Slice'
        score_selector_attrs = {'name': scope.get_unique_operator_name(score_selector_type)}
        score_selector_attrs['starts'] = [0, 1]
        score_selector_attrs['ends'] = [1, 2]
        container.add_node(score_selector_type, normalized_probability_tensor_name, operator.outputs[1].full_name,
                           op_version=2, **score_selector_attrs)


register_converter('SklearnLinearClassifier', convert_sklearn_linear_classifier)
register_converter('SklearnLinearSVC', convert_sklearn_linear_classifier)

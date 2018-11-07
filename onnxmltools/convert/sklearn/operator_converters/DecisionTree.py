# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import numbers, six
from ...common._registration import register_converter
from ...common.tree_ensemble import get_default_tree_classifier_attribute_pairs, get_default_tree_regressor_attribute_pairs, add_tree_to_attribute_pairs


def convert_sklearn_decision_tree_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}

    classes = op.classes_
    if all(isinstance(i, np.ndarray) for i in classes):
        classes = np.concatenate(classes)
    if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        attrs['classlabels_int64s'] = class_labels
        zipmap_attrs['classlabels_int64s'] = class_labels
    elif all(isinstance(i, (six.string_types, six.text_type)) for i in classes):
        class_labels = [str(i) for i in classes]
        attrs['classlabels_strings'] = class_labels
        zipmap_attrs['classlabels_strings'] = class_labels
    else:
        raise ValueError('Only support pure string or integer class labels')

    add_tree_to_attribute_pairs(attrs, True, op.tree_, 0, 1., 0, True)

    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name, probability_tensor_name],
                       op_domain='ai.onnx.ml', **attrs)
    container.add_node('ZipMap', probability_tensor_name, operator.outputs[1].full_name,
                       op_domain='ai.onnx.ml', **zipmap_attrs)


def convert_sklearn_decision_tree_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = int(op.n_outputs_)
    add_tree_to_attribute_pairs(attrs, False, op.tree_, 0, 1., 0, False)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnDecisionTreeClassifier', convert_sklearn_decision_tree_classifier)
register_converter('SklearnDecisionTreeRegressor', convert_sklearn_decision_tree_regressor)

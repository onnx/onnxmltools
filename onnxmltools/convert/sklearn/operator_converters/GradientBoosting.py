# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import numbers, six
from ...common._registration import register_converter
from ...common.tree_ensemble import get_default_tree_classifier_attribute_pairs, get_default_tree_regressor_attribute_pairs, add_tree_to_attribute_pairs


def convert_sklearn_gradient_boosting_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}

    if op.n_classes_ == 2:
        transform = 'LOGISTIC'
        base_values = [op.init_.prior]
    else:
        transform = 'SOFTMAX'
        base_values = op.init_.priors
    attrs['base_values'] = [float(v) for v in base_values]
    attrs['post_transform'] = transform

    classes = op.classes_
    if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        attrs['classlabels_int64s'] = class_labels
        zipmap_attrs['classlabels_int64s'] = class_labels
    elif all(isinstance(i, (six.string_types, six.text_type)) for i in classes):
        class_labels = [str(i) for i in classes]
        attrs['classlabels_strings'] = class_labels
        zipmap_attrs['classlabels_strings'] = class_labels
    else:
        raise ValueError('Only string or integer label vector is allowed')

    tree_weight = op.learning_rate
    if op.n_classes_ == 2:
        for tree_id in range(op.n_estimators):
            tree = op.estimators_[tree_id][0].tree_
            add_tree_to_attribute_pairs(attrs, True, tree, tree_id, tree_weight, 0, False)
    else:
        for i in range(op.n_estimators):
            for c in range(op.n_classes_):
                tree_id = i * op.n_classes_ + c
                tree = op.estimators_[i][c].tree_
                add_tree_to_attribute_pairs(attrs, True, tree, tree_id, tree_weight, c, False)

    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name, probability_tensor_name],
                       op_domain='ai.onnx.ml', **attrs)

    container.add_node('ZipMap', probability_tensor_name, operator.outputs[1].full_name, op_domain='ai.onnx.ml',
                       **zipmap_attrs)


def convert_sklearn_gradient_boosting_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'
    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = 1
    attrs['base_values'] = [float(op.init_.mean)]

    tree_weight = op.learning_rate
    for i in range(op.n_estimators):
        tree = op.estimators_[i][0].tree_
        tree_id = i
        add_tree_to_attribute_pairs(attrs, False, tree, tree_id, tree_weight, 0, False)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnGradientBoostingClassifier', convert_sklearn_gradient_boosting_classifier)
register_converter('SklearnGradientBoostingRegressor', convert_sklearn_gradient_boosting_regressor)

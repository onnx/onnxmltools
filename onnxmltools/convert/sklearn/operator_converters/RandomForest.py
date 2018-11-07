# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import numbers, six
from ...common._registration import register_converter
from ...common.tree_ensemble import get_default_tree_classifier_attribute_pairs, get_default_tree_regressor_attribute_pairs, add_tree_to_attribute_pairs


def convert_sklearn_random_forest_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'
    classes = op.classes_

    if all(isinstance(i, np.ndarray) for i in classes):
        classes = np.concatenate(classes)
    attr_pairs = get_default_tree_classifier_attribute_pairs()
    attr_pairs['name'] = scope.get_unique_operator_name(op_type)
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}

    if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        attr_pairs['classlabels_int64s'] = class_labels
        zipmap_attrs['classlabels_int64s'] = class_labels
    elif all(isinstance(i, (six.text_type, six.string_types)) for i in classes):
        class_labels = [str(i) for i in classes]
        attr_pairs['classlabels_strings'] = class_labels
        zipmap_attrs['classlabels_strings'] = class_labels
    else:
        raise ValueError('Only string and integer class labels are allowed')

    # random forest calculate the final score by averaging over all trees' outcomes, so all trees' weights are identical.
    tree_weight = 1. / op.n_estimators

    for tree_id in range(op.n_estimators):
        tree = op.estimators_[tree_id].tree_
        add_tree_to_attribute_pairs(attr_pairs, True, tree, tree_id, tree_weight, 0, True)

    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name, probability_tensor_name],
                       op_domain='ai.onnx.ml', **attr_pairs)

    container.add_node('ZipMap', probability_tensor_name, operator.outputs[1].full_name,
                       op_domain='ai.onnx.ml', **zipmap_attrs)


def convert_sklearn_random_forest_regressor_converter(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'
    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = int(op.n_outputs_)

    # random forest calculate the final score by averaging over all trees' outcomes, so all trees' weights are identical.
    tree_weight = 1. / op.n_estimators
    for tree_id in range(op.n_estimators):
        tree = op.estimators_[tree_id].tree_
        add_tree_to_attribute_pairs(attrs, False, tree, tree_id, tree_weight, 0, False)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnRandomForestClassifier', convert_sklearn_random_forest_classifier)
register_converter('SklearnRandomForestRegressor', convert_sklearn_random_forest_regressor_converter)
register_converter('SklearnExtraTreesClassifier', convert_sklearn_random_forest_classifier)
register_converter('SklearnExtraTreesRegressor', convert_sklearn_random_forest_regressor_converter)

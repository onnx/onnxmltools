# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import numbers, six
from ...common._registration import register_converter


def _get_default_tree_classifier_attribute_pairs():
    attrs = {}
    attrs['post_transform'] = 'NONE'
    attrs['nodes_treeids'] = []
    attrs['nodes_nodeids'] = []
    attrs['nodes_featureids'] = []
    attrs['nodes_modes'] = []
    attrs['nodes_values'] = []
    attrs['nodes_truenodeids'] = []
    attrs['nodes_falsenodeids'] = []
    attrs['nodes_missing_value_tracks_true'] = []
    attrs['nodes_hitrates'] = []
    attrs['class_treeids'] = []
    attrs['class_nodeids'] = []
    attrs['class_ids'] = []
    attrs['class_weights'] = []
    return attrs


def _get_default_tree_regressor_attribute_pairs():
    attrs = {}
    attrs['post_transform'] = 'NONE'
    attrs['n_targets'] = 0
    attrs['nodes_treeids'] = []
    attrs['nodes_nodeids'] = []
    attrs['nodes_featureids'] = []
    attrs['nodes_modes'] = []
    attrs['nodes_values'] = []
    attrs['nodes_truenodeids'] = []
    attrs['nodes_falsenodeids'] = []
    attrs['nodes_missing_value_tracks_true'] = []
    attrs['nodes_hitrates'] = []
    attrs['target_treeids'] = []
    attrs['target_nodeids'] = []
    attrs['target_ids'] = []
    attrs['target_weights'] = []
    return attrs


def _add_node(attr_pairs, is_classifier, tree_id, tree_weight, node_id, feature_id, mode, value, true_child_id,
              false_child_id, weights, weight_id_bias, leaf_weights_are_counts):
    attr_pairs['nodes_treeids'].append(tree_id)
    attr_pairs['nodes_nodeids'].append(node_id)
    attr_pairs['nodes_featureids'].append(feature_id)
    attr_pairs['nodes_modes'].append(mode)
    attr_pairs['nodes_values'].append(value)
    attr_pairs['nodes_truenodeids'].append(true_child_id)
    attr_pairs['nodes_falsenodeids'].append(false_child_id)
    attr_pairs['nodes_missing_value_tracks_true'].append(False)
    attr_pairs['nodes_hitrates'].append(1.)

    # Add leaf information for making prediction
    if mode == 'LEAF':
        flattened_weights = weights.flatten()
        factor = tree_weight
        # If the values stored at leaves are counts of possible classes, we need convert them to probabilities by
        # doing a normalization.
        if leaf_weights_are_counts:
            s = sum(flattened_weights)
            factor /= float(s) if s != 0. else 1.
        flattened_weights = [w * factor for w in flattened_weights]
        if len(flattened_weights) == 2 and is_classifier:
            flattened_weights = [flattened_weights[1]]

        # Note that attribute names for making prediction are different for classifiers and regressors
        if is_classifier:
            for i, w in enumerate(flattened_weights):
                attr_pairs['class_treeids'].append(tree_id)
                attr_pairs['class_nodeids'].append(node_id)
                attr_pairs['class_ids'].append(i + weight_id_bias)
                attr_pairs['class_weights'].append(w)
        else:
            for i, w in enumerate(flattened_weights):
                attr_pairs['target_treeids'].append(tree_id)
                attr_pairs['target_nodeids'].append(node_id)
                attr_pairs['target_ids'].append(i + weight_id_bias)
                attr_pairs['target_weights'].append(w)


def _add_tree_to_attribute_pairs(attr_pairs, is_classifier, tree, tree_id, tree_weight,
                                 weight_id_bias, leaf_weights_are_counts):
    for i in range(tree.node_count):
        node_id = i
        weight = tree.value[i]

        if tree.children_left[i] > i or tree.children_right[i] > i:
            mode = 'BRANCH_LEQ'
            feat_id = tree.feature[i]
            threshold = tree.threshold[i]
            left_child_id = int(tree.children_left[i])
            right_child_id = int(tree.children_right[i])
        else:
            mode = 'LEAF'
            feat_id = 0
            threshold = 0.
            left_child_id = 0
            right_child_id = 0

        _add_node(attr_pairs, is_classifier, tree_id, tree_weight, node_id, feat_id, mode, threshold,
                  left_child_id, right_child_id, weight, weight_id_bias, leaf_weights_are_counts)


def convert_sklearn_decision_tree_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attrs = _get_default_tree_classifier_attribute_pairs()
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

    _add_tree_to_attribute_pairs(attrs, True, op.tree_, 0, 1., 0, True)

    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name, probability_tensor_name],
                       op_domain='ai.onnx.ml', **attrs)
    container.add_node('ZipMap', probability_tensor_name, operator.outputs[1].full_name,
                       op_domain='ai.onnx.ml', **zipmap_attrs)


def convert_sklearn_decision_tree_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'

    attrs = _get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = op.n_outputs_
    _add_tree_to_attribute_pairs(attrs, False, op.tree_, 0, 1., 0, False)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


def convert_sklearn_random_forest_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'
    classes = op.classes_

    if all(isinstance(i, np.ndarray) for i in classes):
        classes = np.concatenate(classes)
    attr_pairs = _get_default_tree_classifier_attribute_pairs()
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
        _add_tree_to_attribute_pairs(attr_pairs, True, tree, tree_id, tree_weight, 0, True)

    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name, probability_tensor_name],
                       op_domain='ai.onnx.ml', **attr_pairs)

    container.add_node('ZipMap', probability_tensor_name, operator.outputs[1].full_name,
                       op_domain='ai.onnx.ml', **zipmap_attrs)


def convert_sklearn_random_forest_regressor_converter(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'
    attrs = _get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = op.n_outputs_

    # random forest calculate the final score by averaging over all trees' outcomes, so all trees' weights are identical.
    tree_weight = 1. / op.n_estimators
    for tree_id in range(op.n_estimators):
        tree = op.estimators_[tree_id].tree_
        _add_tree_to_attribute_pairs(attrs, False, tree, tree_id, tree_weight, 0, False)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


def convert_sklearn_gradient_boosting_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attrs = _get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}

    if op.n_classes_ == 2:
        transform = 'LOGISTIC'
        base_values = [op.init_.prior]
    else:
        transform = 'SOFTMAX'
        base_values = op.init_.priors
    attrs['base_values'] = base_values
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
            _add_tree_to_attribute_pairs(attrs, True, tree, tree_id, tree_weight, 0, False)
    else:
        for i in range(op.n_estimators):
            for c in range(op.n_classes_):
                tree_id = i * op.n_classes_ + c
                tree = op.estimators_[i][c].tree_
                _add_tree_to_attribute_pairs(attrs, True, tree, tree_id, tree_weight, c, False)

    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name, probability_tensor_name],
                       op_domain='ai.onnx.ml', **attrs)

    container.add_node('ZipMap', probability_tensor_name, operator.outputs[1].full_name, op_domain='ai.onnx.ml',
                       **zipmap_attrs)


def convert_sklearn_gradient_boosting_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'
    attrs = _get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = 1
    attrs['base_values'] = [op.init_.mean]

    tree_weight = op.learning_rate
    for i in range(op.n_estimators):
        tree = op.estimators_[i][0].tree_
        tree_id = i
        _add_tree_to_attribute_pairs(attrs, False, tree, tree_id, tree_weight, 0, False)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnDecisionTreeClassifier', convert_sklearn_decision_tree_classifier)
register_converter('SklearnDecisionTreeRegressor', convert_sklearn_decision_tree_regressor)
register_converter('SklearnRandomForestClassifier', convert_sklearn_random_forest_classifier)
register_converter('SklearnRandomForestRegressor', convert_sklearn_random_forest_regressor_converter)
register_converter('SklearnExtraTreesClassifier', convert_sklearn_random_forest_classifier)
register_converter('SklearnExtraTreesRegressor', convert_sklearn_random_forest_regressor_converter)
register_converter('SklearnGradientBoostingClassifier', convert_sklearn_gradient_boosting_classifier)
register_converter('SklearnGradientBoostingRegressor', convert_sklearn_gradient_boosting_regressor)

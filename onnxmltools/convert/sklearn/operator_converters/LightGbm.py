# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers, six
import numpy as np
from ...common._registration import register_converter


def _get_default_attributes():
    attrs = {}
    # Node attributes
    attrs['nodes_treeids'] = []
    attrs['nodes_nodeids'] = []
    attrs['nodes_featureids'] = []
    attrs['nodes_modes'] = []
    attrs['nodes_values'] = []
    attrs['nodes_truenodeids'] = []
    attrs['nodes_falsenodeids'] = []
    attrs['nodes_hitrates'] = []
    attrs['nodes_missing_value_tracks_true'] = []

    # Leaf attributes
    attrs['class_ids'] = []
    attrs['class_nodeids'] = []
    attrs['class_treeids'] = []
    attrs['class_weights'] = []
    return attrs


def _translate_split_criterion(criterion):
    # If the criterion is true, LightGBM use the left child. Otherwise, right child is selected.
    if criterion == '<=':
        return 'BRANCH_LEQ'
    elif criterion == '<':
        return 'BRANCH_LT'
    elif criterion == '>=':
        return 'BRANCH_GTE'
    elif criterion == '>':
        return 'BRANCH_GT'
    else:
        raise ValueError('Unsupported splitting criterion: %s. Only <=, <, >=, and > are allowed.')


def _create_node_id(node_id_pool):
    i = 0
    while i in node_id_pool:
        i += 1
    node_id_pool.add(i)
    return i


def _parse_tree_structure(tree_id, class_id, learning_rate, tree_structure, attrs):
    # The pool of all nodes' indexes created when parsing a single tree. Different trees may use different pools.
    node_id_pool = set()

    node_id = _create_node_id(node_id_pool)
    left_id = _create_node_id(node_id_pool)
    right_id = _create_node_id(node_id_pool)

    attrs['nodes_treeids'].append(tree_id)
    attrs['nodes_nodeids'].append(node_id)

    attrs['nodes_featureids'].append(tree_structure['split_feature'])
    attrs['nodes_modes'].append(_translate_split_criterion(tree_structure['decision_type']))
    attrs['nodes_values'].append(tree_structure['threshold'])

    # Assume left is the true branch and right is the false branch
    attrs['nodes_truenodeids'].append(left_id)
    attrs['nodes_falsenodeids'].append(right_id)
    attrs['nodes_hitrates'].append(1.)
    if tree_structure['default_left']:
        attrs['nodes_missing_value_tracks_true'].append(1)
    else:
        attrs['nodes_missing_value_tracks_true'].append(0)
    _parse_node(tree_id, class_id, left_id, node_id_pool, learning_rate, tree_structure['left_child'], attrs)
    _parse_node(tree_id, class_id, right_id, node_id_pool, learning_rate, tree_structure['right_child'], attrs)


def _parse_node(tree_id, class_id, node_id, node_id_pool, learning_rate, node, attrs):
    if (hasattr(node, 'left_child') and hasattr(node, 'right_child')) or \
            ('left_child' in node and 'right_child' in node):
        left_id = _create_node_id(node_id_pool)
        right_id = _create_node_id(node_id_pool)

        attrs['nodes_treeids'].append(tree_id)
        attrs['nodes_nodeids'].append(node_id)

        attrs['nodes_featureids'].append(node['split_feature'])
        attrs['nodes_modes'].append(_translate_split_criterion(node['decision_type']))
        attrs['nodes_values'].append(node['threshold'])

        # Assume left is the true branch and right is the false branch
        attrs['nodes_truenodeids'].append(left_id)
        attrs['nodes_falsenodeids'].append(right_id)
        attrs['nodes_hitrates'].append(1.)
        if node['default_left']:
            attrs['nodes_missing_value_tracks_true'].append(1)
        else:
            attrs['nodes_missing_value_tracks_true'].append(0)

        # Recursively dive into the child nodes
        _parse_node(tree_id, class_id, left_id, node_id_pool, learning_rate, node['left_child'], attrs)
        _parse_node(tree_id, class_id, right_id, node_id_pool, learning_rate, node['right_child'], attrs)
    elif hasattr(node, 'left_child') or hasattr(node, 'right_child'):
        raise ValueError('Need two branches')
    else:
        # Node attributes
        attrs['nodes_treeids'].append(tree_id)
        attrs['nodes_nodeids'].append(node_id)
        attrs['nodes_featureids'].append(0)
        attrs['nodes_modes'].append('LEAF')
        # Leaf node has no threshold. A zero is appended but it will never be used.
        attrs['nodes_values'].append(0.)
        # Leaf node has no child. A zero is appended but it will never be used.
        attrs['nodes_truenodeids'].append(0)
        # Leaf node has no child. A zero is appended but it will never be used.
        attrs['nodes_falsenodeids'].append(0)
        # This attribute is not used, so we assign it a constant value.
        attrs['nodes_hitrates'].append(1.)
        # Leaf node has no split function. A zero is appended but it will never be used.
        attrs['nodes_missing_value_tracks_true'].append(0)

        # Leaf attributes
        attrs['class_treeids'].append(tree_id)
        attrs['class_nodeids'].append(node_id)
        attrs['class_ids'].append(class_id)
        attrs['class_weights'].append(node['leaf_value'] * learning_rate)


def convert_lightgbm_classifier(scope, operator, container):
    gbm_model = operator.raw_operator
    gbm_text = gbm_model.booster_.dump_model()

    attrs = _get_default_attributes()
    attrs['name'] = operator.full_name
    attrs['post_transform'] = 'NONE'
    zipmap_attrs = {'name': scope.get_unique_variable_name('ZipMap')}
    n_classes = gbm_text['num_class']
    for tree in gbm_text['tree_info']:
        for class_id in range(n_classes):
            tree_id = tree['tree_index']
            learning_rate = tree['shrinkage']
            _parse_tree_structure(tree_id, class_id, learning_rate, tree['tree_structure'], attrs)

    if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in gbm_model.classes_):
        class_labels = [int(i) for i in gbm_model.classes_]
        attrs['classlabels_int64s'] = class_labels
        zipmap_attrs['classlabels_int64s'] = class_labels
    elif all(isinstance(i, (six.text_type, six.string_types)) for i in gbm_model.classes_):
        class_labels = [str(i) for i in gbm_model.classes_]
        attrs['classlabels_strings'] = class_labels
        zipmap_attrs['classlabels_strings'] = class_labels
    else:
        raise ValueError('Only string and integer class labels are allowed')

    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
    container.add_node('TreeEnsembleClassifier', operator.input_full_names,
                       [operator.outputs[0].full_name, probability_tensor_name],
                       op_domain='ai.onnx.ml', **attrs)

    container.add_node('ZipMap', probability_tensor_name, operator.outputs[1].full_name,
                       op_domain='ai.onnx.ml', **zipmap_attrs)


register_converter('LgbmClassifier', convert_lightgbm_classifier)

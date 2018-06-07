# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
import numbers, six
import numpy as np
from collections import Counter
from lightgbm import LGBMClassifier, LGBMRegressor
from ...common._registration import register_converter
from .TreeEnsemble import _get_default_tree_classifier_attribute_pairs


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
    if tree_structure['default_left']:
        attrs['nodes_missing_value_tracks_true'].append(1)
    else:
        attrs['nodes_missing_value_tracks_true'].append(0)
    attrs['nodes_hitrates'].append(1.)
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
        if node['default_left']:
            attrs['nodes_missing_value_tracks_true'].append(1)
        else:
            attrs['nodes_missing_value_tracks_true'].append(0)
        attrs['nodes_hitrates'].append(1.)

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
        # Leaf node has no split function. A zero is appended but it will never be used.
        attrs['nodes_missing_value_tracks_true'].append(0)
        attrs['nodes_hitrates'].append(1.)

        # Leaf attributes
        attrs['class_treeids'].append(tree_id)
        attrs['class_nodeids'].append(node_id)
        attrs['class_ids'].append(class_id)
        attrs['class_weights'].append(float(node['leaf_value']) * learning_rate)


def convert_lightgbm(scope, operator, container):
    gbm_model = operator.raw_operator
    if gbm_model.boosting_type != 'gbdt':
        raise ValueError('Only support LightGBM classifier with boosting_type=gbdt')
    gbm_text = gbm_model.booster_.dump_model()

    attrs = _get_default_tree_classifier_attribute_pairs()
    attrs['name'] = operator.full_name

    # Create different attributes for classifier and regressor, respectively
    if isinstance(gbm_model, LGBMClassifier):
        n_classes = gbm_text['num_class']
        if gbm_model.objective_ == 'multiclass':
            attrs['post_transform'] = 'SOFTMAX'
        else:
            attrs['post_transform'] = 'LOGISTIC'
    else:
        n_classes = 1  # Regressor has only one output variable
        attrs['post_transform'] = 'NONE'

    # Use the same algorithm to parse the tree
    for i, tree in enumerate(gbm_text['tree_info']):
        tree_id = i
        class_id = tree_id % n_classes
        learning_rate = tree['shrinkage']
        _parse_tree_structure(tree_id, class_id, learning_rate, tree['tree_structure'], attrs)

    # Sort nodes_* attributes. For one tree, its node indexes should appear in an ascent order in nodes_nodeids. Nodes
    # from a tree with a smaller tree index should appear before trees with larger indexes in nodes_nodeids.
    node_numbers_per_tree = Counter(attrs['nodes_treeids'])
    tree_number = len(node_numbers_per_tree.keys())
    accumulated_node_numbers = [0] * tree_number
    for i in range(1, tree_number):
        accumulated_node_numbers[i] = accumulated_node_numbers[i - 1] + node_numbers_per_tree[i - 1]
    global_node_indexes = []
    for i in range(len(attrs['nodes_nodeids'])):
        tree_id = attrs['nodes_treeids'][i]
        node_id = attrs['nodes_nodeids'][i]
        global_node_indexes.append(accumulated_node_numbers[tree_id] + node_id)
    for k, v in attrs.items():
        if k.startswith('nodes_'):
            merged_indexes = zip(copy.deepcopy(global_node_indexes), v)
            sorted_list = [pair[1] for pair in sorted(merged_indexes, key=lambda x: x[0])]
            attrs[k] = sorted_list

    # Create ONNX object
    if isinstance(gbm_model, LGBMClassifier):
        # Prepare label information for both of TreeEnsembleClassifier and ZipMap
        zipmap_attrs = {'name': scope.get_unique_variable_name('ZipMap')}
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

        # Create tree classifier
        probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
        container.add_node('TreeEnsembleClassifier', operator.input_full_names,
                           [operator.outputs[0].full_name, probability_tensor_name],
                           op_domain='ai.onnx.ml', **attrs)

        # Convert probability tensor to probability map (keys are labels while values are the associated probabilities)
        container.add_node('ZipMap', probability_tensor_name, operator.outputs[1].full_name,
                           op_domain='ai.onnx.ml', **zipmap_attrs)
    else:
        # Create tree regressor
        keys_to_be_renamed = list(k for k in attrs.keys() if k.startswith('class_'))
        for k in keys_to_be_renamed:
            # Rename class_* attribute to target_* because TreeEnsebmleClassifier and TreeEnsembleClassifier have
            # different ONNX attributes
            attrs['target' + k[5:]] = copy.deepcopy(attrs[k])
            del attrs[k]
        container.add_node('TreeEnsembleRegressor', operator.input_full_names,
                           operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('LgbmClassifier', convert_lightgbm)
register_converter('LgbmRegressor', convert_lightgbm)

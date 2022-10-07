# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator
from .tree_ensemble_common import save_read_sparkml_model_data, \
    sparkml_tree_dataset_to_sklearn
from .tree_helper import Node

logger = logging.getLogger("onnxmltools")


def get_default_tree_classifier_attribute_pairs():
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


def get_default_tree_regressor_attribute_pairs():
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


def add_node(attr_pairs, is_classifier, tree_id, tree_weight, node_id, feature_id, mode, value, true_child_id,
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


def add_tree_to_attribute_pairs(attr_pairs, is_classifier, tree, tree_id, tree_weight,
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

        add_node(attr_pairs, is_classifier, tree_id, tree_weight, node_id, feat_id, mode, threshold,
                 left_child_id, right_child_id, weight, weight_id_bias, leaf_weights_are_counts)


def convert_decision_tree_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs["classlabels_int64s"] = list(range(0, op.numClasses))

    logger.info("[convert_decision_tree_classifier] save_read_sparkml_model_data")
    tree_df = save_read_sparkml_model_data(operator.raw_params['SparkSession'], op)
    logger.info("[convert_decision_tree_classifier] sparkml_tree_dataset_to_sklearn")
    tree = sparkml_tree_dataset_to_sklearn(tree_df, is_classifier=True)
    logger.info("[convert_decision_tree_classifier] add_tree_to_attribute_pairs")
    add_tree_to_attribute_pairs(attrs, True, tree, 0, 1., 0, leaf_weights_are_counts=True)
    logger.info("[convert_decision_tree_classifier] n_nodes=%d", len(attrs['nodes_nodeids']))

    # Some values appear in an array of one element instead of a float.
    in_sets_rules = []
    for i, value in enumerate(attrs["nodes_values"]):
        if isinstance(value, (np.ndarray, list)):
            in_sets_rules.append(i)
    if True or len(in_sets_rules) > 0:
        logger.info("[convert_decision_tree_classifier] in_set_rules has %d elements", len(in_sets_rules))
        for i in in_sets_rules:
            attrs["nodes_modes"][i] = "||"
        logger.info("[convert_decision_tree_classifier] Node.create")
        root, _ = Node.create(attrs)
        logger.info("[convert_decision_tree_classifier] unfold_rule_or")
        root.unfold_rule_or()
        logger.info("[convert_decision_tree_classifier] to_attrs")
        new_attrs = root.to_attrs(
                post_transform=attrs['post_transform'],
                classlabels_int64s=attrs["classlabels_int64s"])
        attrs = new_attrs
        logger.info("[convert_decision_tree_classifier] n_nodes=%d", len(attrs['nodes_nodeids']))

    logger.info("[convert_decision_tree_classifier] end")

    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name,
                       operator.outputs[1].full_name], op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.classification.DecisionTreeClassificationModel', convert_decision_tree_classifier)


def calculate_decision_tree_classifier_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]

    class_count = operator.raw_operator.numClasses
    operator.outputs[0].type = Int64TensorType(shape=[N])
    operator.outputs[1].type = FloatTensorType([N, class_count])


register_shape_calculator('pyspark.ml.classification.DecisionTreeClassificationModel',
                          calculate_decision_tree_classifier_output_shapes)

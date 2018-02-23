#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...proto import onnx_proto
from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from ..common import model_util
from .common import add_zipmap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor



def _get_default_tree_classifier_attribute_pairs():
    attrs = {}
    attrs['post_transform'] = 'NONE'
    attrs['classlabels_strings'] = []
    attrs['classlabels_int64s'] = []
    attrs['nodes_treeids'] = []
    attrs['nodes_nodeids'] = []
    attrs['nodes_featureids'] = []
    attrs['nodes_modes'] = []
    attrs['nodes_values'] = []
    attrs['nodes_truenodeids'] = []
    attrs['nodes_falsenodeids'] = []
    attrs['nodes_missing_value_tracks_true'] = []
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


class DecisionTreeClassifierConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'tree_')
            utils._check_has_attr(sk_node, 'classes_')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in sklearn model ' + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        attr_pairs = _get_default_tree_classifier_attribute_pairs()
        classes = sk_node.classes_
        if utils.is_numeric_type(sk_node.classes_):
            class_labels = utils.cast_list(int, classes)
            attr_pairs['classlabels_int64s'] = class_labels
            output_type = onnx_proto.TensorProto.INT64
        else:
            class_labels = utils.cast_list(str, classes)
            attr_pairs['classlabels_strings'] = class_labels
            output_type = onnx_proto.TensorProto.STRING

        _add_tree_to_attribute_pairs(attr_pairs, True, sk_node.tree_, 0, 1., 0, True)
        nb = NodeBuilder(context, "TreeEnsembleClassifier")

        for k, v in attr_pairs.items():
            if isinstance(v, list) and len(v) == 0:
                continue
            nb.add_attribute(k, v)

        nb.extend_inputs(inputs)
        output_dim = [1]
        
        output_y = model_util.make_tensor_value_info(nb.name + '.Y', output_type, output_dim)
        nb.add_output(output_y)
        context.add_output(output_y)
        
        prob_input = context.get_unique_name('classProbability')
        nb.add_output(prob_input)
        appended_node = add_zipmap(prob_input, output_type, class_labels, context)

        return [nb.make_node(), appended_node]


class DecisionTreeRegressorConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'tree_')
            utils._check_has_attr(sk_node, 'n_outputs_')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in sklearn model')

    @staticmethod
    def convert(context, sk_node, inputs):
        attr_pairs = _get_default_tree_regressor_attribute_pairs()

        attr_pairs['n_targets'] = sk_node.n_outputs_

        _add_tree_to_attribute_pairs(attr_pairs, False, sk_node.tree_, 0, 1., 0, False)

        nb = NodeBuilder(context, "TreeEnsembleRegressor")

        for k, v in attr_pairs.items():
            nb.add_attribute(k, v)

        nb.extend_inputs(inputs)
        output_dim = [1,sk_node.n_outputs_]
        nb.add_output(model_util.make_tensor_value_info(nb.name, onnx_proto.TensorProto.FLOAT, output_dim))

        return nb.make_node()


class RandomForestClassifierConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'n_classes_')
            utils._check_has_attr(sk_node, 'classes_')
            utils._check_has_attr(sk_node, 'n_estimators')
            utils._check_has_attr(sk_node, 'estimators_')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in sklearn model')

    @staticmethod
    def convert(context, sk_node, inputs):
        classes = sk_node.classes_
        attr_pairs = _get_default_tree_classifier_attribute_pairs()
        if utils.is_numeric_type(sk_node.classes_):
            class_labels = utils.cast_list(int, classes)
            attr_pairs['classlabels_int64s'] = class_labels
            output_type = onnx_proto.TensorProto.INT64
        else:
            class_labels = utils.cast_list(str, classes)
            attr_pairs['classlabels_strings'] = class_labels
            output_type = onnx_proto.TensorProto.STRING

        tree_weight = 1. / sk_node.n_estimators
        for i in range(sk_node.n_estimators):
            tree = sk_node.estimators_[i].tree_
            tree_id = i
            _add_tree_to_attribute_pairs(attr_pairs, True, tree, tree_id, tree_weight, 0, True)

        nb = NodeBuilder(context, "TreeEnsembleClassifier")

        for k, v in attr_pairs.items():
            if isinstance(v, list) and len(v) == 0:
                continue
            nb.add_attribute(k, v)

        nb.extend_inputs(inputs)
        output_dim = [1]

        output_y = model_util.make_tensor_value_info(nb.name + '.Y', output_type, output_dim)
        nb.add_output(output_y)
        context.add_output(output_y)

        prob_input = context.get_unique_name('classProbability')
        nb.add_output(prob_input)
        appended_node = add_zipmap(prob_input, output_type, class_labels, context)
        
        return [nb.make_node(), appended_node]


class RandomForestRegressorConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'n_outputs_')
            utils._check_has_attr(sk_node, 'n_estimators')
            utils._check_has_attr(sk_node, 'estimators_')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in sklearn model')

    @staticmethod
    def convert(context, sk_node, inputs):
        attr_pairs = _get_default_tree_regressor_attribute_pairs()

        attr_pairs['n_targets'] = sk_node.n_outputs_

        tree_weight = 1. / sk_node.n_estimators
        for i in range(sk_node.n_estimators):
            tree = sk_node.estimators_[i].tree_
            tree_id = i
            _add_tree_to_attribute_pairs(attr_pairs, False, tree, tree_id, tree_weight, 0, False)

        nb = NodeBuilder(context, "TreeEnsembleRegressor")

        for k, v in attr_pairs.items():
            nb.add_attribute(k, v)

        nb.extend_inputs(inputs)
        output_dim = [1, sk_node.n_outputs_]
        nb.add_output(model_util.make_tensor_value_info(nb.name, onnx_proto.TensorProto.FLOAT, output_dim))

        return nb.make_node()


class GradientBoostingClassifierConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'n_classes_')
            utils._check_has_attr(sk_node, 'classes_')
            utils._check_has_attr(sk_node, 'n_estimators')
            utils._check_has_attr(sk_node, 'estimators_')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in sklearn model')

    @staticmethod
    def convert(context, sk_node, inputs):
        attr_pairs = _get_default_tree_classifier_attribute_pairs()
        classes = sk_node.classes_

        if sk_node.n_classes_ == 2:
            transform = 'LOGISTIC'
            base_values = [sk_node.init_.prior]
        else:
            transform = 'SOFTMAX'
            base_values = sk_node.init_.priors
        attr_pairs['base_values'] = base_values
        attr_pairs['post_transform'] = transform

        if utils.is_numeric_type(classes):
            class_labels = utils.cast_list(int, classes)
            attr_pairs['classlabels_int64s'] = class_labels
            output_type = onnx_proto.TensorProto.INT64
        else:
            class_labels = utils.cast_list(str, classes)
            attr_pairs['classlabels_strings'] = class_labels
            output_type = onnx_proto.TensorProto.STRING

        tree_weight = sk_node.learning_rate
        if sk_node.n_classes_ == 2:
            for i in range(sk_node.n_estimators):
                tree_id = i
                tree = sk_node.estimators_[i][0].tree_
                _add_tree_to_attribute_pairs( attr_pairs, True, tree, tree_id, tree_weight, 0, False)
        else:
            for i in range(sk_node.n_estimators):
                for c in range(sk_node.n_classes_):
                    tree_id = i * sk_node.n_classes_ + c
                    tree = sk_node.estimators_[i][c].tree_
                    _add_tree_to_attribute_pairs(attr_pairs, True, tree, tree_id, tree_weight, c, False)

        nb = NodeBuilder(context, "TreeEnsembleClassifier")

        for k, v in attr_pairs.items():
            if isinstance(v, list) and len(v) == 0:
                continue
            nb.add_attribute(k, v)

        nb.extend_inputs(inputs)
        output_dim = [1]
        
        output_y = model_util.make_tensor_value_info(nb.name + '.Y', output_type, output_dim)
        nb.add_output(output_y)
        context.add_output(output_y)

        prob_input = context.get_unique_name('classProbability')
        nb.add_output(prob_input)
        appended_node = add_zipmap(prob_input, output_type, class_labels, context)

        return [nb.make_node(), appended_node]


class GradientBoostingRegressorConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'n_estimators')
            utils._check_has_attr(sk_node, 'estimators_')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in sklearn model')

    @staticmethod
    def convert(context, sk_node, inputs):
        attr_pairs = _get_default_tree_regressor_attribute_pairs()
        attr_pairs['n_targets'] = 1
        attr_pairs['base_values'] = [utils.convert_to_python_value(sk_node.init_.mean)]

        tree_weight = sk_node.learning_rate
        for i in range(sk_node.n_estimators):
            tree = sk_node.estimators_[i][0].tree_
            tree_id = i
            _add_tree_to_attribute_pairs(attr_pairs, False, tree, tree_id, tree_weight, 0, False)

        nb = NodeBuilder(context, "TreeEnsembleRegressor")

        for k, v in attr_pairs.items():
            nb.add_attribute(k, v)

        nb.extend_inputs(inputs)
        output_dim = [1, 1]
        nb.add_output(model_util.make_tensor_value_info(nb.name, onnx_proto.TensorProto.FLOAT, output_dim))

        return nb.make_node()


register_converter(DecisionTreeClassifier, DecisionTreeClassifierConverter)
register_converter(DecisionTreeRegressor, DecisionTreeRegressorConverter)
register_converter(RandomForestClassifier, RandomForestClassifierConverter)
register_converter(RandomForestRegressor, RandomForestRegressorConverter)
register_converter(GradientBoostingClassifier, GradientBoostingClassifierConverter)
register_converter(GradientBoostingRegressor, GradientBoostingRegressorConverter)

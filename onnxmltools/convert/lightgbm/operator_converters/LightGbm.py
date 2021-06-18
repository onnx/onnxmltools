# SPDX-License-Identifier: Apache-2.0

import copy
import numbers
import numpy as np
import onnx
from collections import Counter
from ...common._apply_operation import (
    apply_div, apply_reshape, apply_sub, apply_cast, apply_identity, apply_clip)
from ...common._registration import register_converter
from ...common.tree_ensemble import get_default_tree_classifier_attribute_pairs
from ....proto import onnx_proto
from onnxconverter_common.container import ModelComponentContainer


def _translate_split_criterion(criterion):
    # If the criterion is true, LightGBM use the left child.
    # Otherwise, right child is selected.
    if criterion == '<=':
        return 'BRANCH_LEQ'
    elif criterion == '<':
        return 'BRANCH_LT'
    elif criterion == '>=':
        return 'BRANCH_GTE'
    elif criterion == '>':
        return 'BRANCH_GT'
    elif criterion == '==':
        return 'BRANCH_EQ'
    elif criterion == '!=':
        return 'BRANCH_NEQ'
    else:
        raise ValueError(
            'Unsupported splitting criterion: %s. Only <=, '
            '<, >=, and > are allowed.')


def _create_node_id(node_id_pool):
    i = 0
    while i in node_id_pool:
        i += 1
    node_id_pool.add(i)
    return i


def _parse_tree_structure(tree_id, class_id, learning_rate,
                          tree_structure, attrs):
    """
    The pool of all nodes' indexes created when parsing a single tree.
    Different tree use different pools.
    """
    node_id_pool = set()
    node_pyid_pool = dict()

    node_id = _create_node_id(node_id_pool)
    node_pyid_pool[id(tree_structure)] = node_id

    # The root node is a leaf node.
    if ('left_child' not in tree_structure or
            'right_child' not in tree_structure):
        _parse_node(tree_id, class_id, node_id, node_id_pool, node_pyid_pool,
                    learning_rate, tree_structure, attrs)
        return

    left_pyid = id(tree_structure['left_child'])
    right_pyid = id(tree_structure['right_child'])

    if left_pyid in node_pyid_pool:
        left_id = node_pyid_pool[left_pyid]
        left_parse = False
    else:
        left_id = _create_node_id(node_id_pool)
        node_pyid_pool[left_pyid] = left_id
        left_parse = True

    if right_pyid in node_pyid_pool:
        right_id = node_pyid_pool[right_pyid]
        right_parse = False
    else:
        right_id = _create_node_id(node_id_pool)
        node_pyid_pool[right_pyid] = right_id
        right_parse = True

    attrs['nodes_treeids'].append(tree_id)
    attrs['nodes_nodeids'].append(node_id)

    attrs['nodes_featureids'].append(tree_structure['split_feature'])
    attrs['nodes_modes'].append(
        _translate_split_criterion(tree_structure['decision_type']))
    if isinstance(tree_structure['threshold'], str):
        try:
            attrs['nodes_values'].append(float(tree_structure['threshold']))
        except ValueError:
            import pprint
            text = pprint.pformat(tree_structure)
            if len(text) > 100000:
                text = text[:100000] + "\n..."
            raise TypeError("threshold must be a number not '{}'"
                            "\n{}".format(tree_structure['threshold'], text))
    else:
        attrs['nodes_values'].append(tree_structure['threshold'])

    # Assume left is the true branch and right is the false branch
    attrs['nodes_truenodeids'].append(left_id)
    attrs['nodes_falsenodeids'].append(right_id)
    if tree_structure['default_left']:
        attrs['nodes_missing_value_tracks_true'].append(1)
    else:
        attrs['nodes_missing_value_tracks_true'].append(0)
    attrs['nodes_hitrates'].append(1.)
    if left_parse:
        _parse_node(
            tree_id, class_id, left_id, node_id_pool, node_pyid_pool,
            learning_rate, tree_structure['left_child'], attrs)
    if right_parse:
        _parse_node(
            tree_id, class_id, right_id, node_id_pool, node_pyid_pool,
            learning_rate, tree_structure['right_child'], attrs)


def _parse_node(tree_id, class_id, node_id, node_id_pool, node_pyid_pool,
                learning_rate, node, attrs):
    """
    Parses nodes.
    """
    if ((hasattr(node, 'left_child') and hasattr(node, 'right_child')) or
            ('left_child' in node and 'right_child' in node)):

        left_pyid = id(node['left_child'])
        right_pyid = id(node['right_child'])

        if left_pyid in node_pyid_pool:
            left_id = node_pyid_pool[left_pyid]
            left_parse = False
        else:
            left_id = _create_node_id(node_id_pool)
            node_pyid_pool[left_pyid] = left_id
            left_parse = True

        if right_pyid in node_pyid_pool:
            right_id = node_pyid_pool[right_pyid]
            right_parse = False
        else:
            right_id = _create_node_id(node_id_pool)
            node_pyid_pool[right_pyid] = right_id
            right_parse = True

        attrs['nodes_treeids'].append(tree_id)
        attrs['nodes_nodeids'].append(node_id)

        attrs['nodes_featureids'].append(node['split_feature'])
        attrs['nodes_modes'].append(
            _translate_split_criterion(node['decision_type']))
        if isinstance(node['threshold'], str):
            try:
                attrs['nodes_values'].append(float(node['threshold']))
            except ValueError:
                import pprint
                text = pprint.pformat(node)
                if len(text) > 100000:
                    text = text[:100000] + "\n..."
                raise TypeError("threshold must be a number not '{}'"
                                "\n{}".format(node['threshold'], text))
        else:
            attrs['nodes_values'].append(node['threshold'])

        # Assume left is the true branch
        # and right is the false branch
        attrs['nodes_truenodeids'].append(left_id)
        attrs['nodes_falsenodeids'].append(right_id)
        if node['default_left']:
            attrs['nodes_missing_value_tracks_true'].append(1)
        else:
            attrs['nodes_missing_value_tracks_true'].append(0)
        attrs['nodes_hitrates'].append(1.)

        # Recursively dive into the child nodes
        if left_parse:
            _parse_node(
                tree_id, class_id, left_id, node_id_pool, node_pyid_pool,
                learning_rate, node['left_child'], attrs)
        if right_parse:
            _parse_node(
                tree_id, class_id, right_id, node_id_pool, node_pyid_pool,
                learning_rate, node['right_child'], attrs)
    elif hasattr(node, 'left_child') or hasattr(node, 'right_child'):
        raise ValueError('Need two branches')
    else:
        # Node attributes
        attrs['nodes_treeids'].append(tree_id)
        attrs['nodes_nodeids'].append(node_id)
        attrs['nodes_featureids'].append(0)
        attrs['nodes_modes'].append('LEAF')
        # Leaf node has no threshold.
        # A zero is appended but it will never be used.
        attrs['nodes_values'].append(0.)
        # Leaf node has no child.
        # A zero is appended but it will never be used.
        attrs['nodes_truenodeids'].append(0)
        # Leaf node has no child.
        # A zero is appended but it will never be used.
        attrs['nodes_falsenodeids'].append(0)
        # Leaf node has no split function.
        # A zero is appended but it will never be used.
        attrs['nodes_missing_value_tracks_true'].append(0)
        attrs['nodes_hitrates'].append(1.)

        # Leaf attributes
        attrs['class_treeids'].append(tree_id)
        attrs['class_nodeids'].append(node_id)
        attrs['class_ids'].append(class_id)
        attrs['class_weights'].append(
            float(node['leaf_value']) * learning_rate)


def convert_lightgbm(scope, operator, container):
    """
    Converters for *lightgbm*.
    """
    gbm_model = operator.raw_operator
    gbm_text = gbm_model.booster_.dump_model()
    modify_tree_for_rule_in_set(gbm_text, use_float=True)

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = operator.full_name

    # Create different attributes for classifier and
    # regressor, respectively
    post_transform = None
    if gbm_text['objective'].startswith('binary'):
        n_classes = 1
        attrs['post_transform'] = 'LOGISTIC'
    elif gbm_text['objective'].startswith('multiclass'):
        n_classes = gbm_text['num_class']
        attrs['post_transform'] = 'SOFTMAX'
    elif gbm_text['objective'].startswith('regression'):
        n_classes = 1  # Regressor has only one output variable
        attrs['post_transform'] = 'NONE'
        attrs['n_targets'] = n_classes
    elif gbm_text['objective'].startswith('poisson'):
        n_classes = 1  # Regressor has only one output variable
        attrs['n_targets'] = n_classes
        # 'Exp' is not a supported post_transform value in the ONNX spec yet,
        # so we need to add an 'Exp' post transform node to the model
        attrs['post_transform'] = 'NONE'
        post_transform = "Exp"
    else:
        raise RuntimeError(
            "LightGBM objective should be cleaned already not '{}'.".format(
                gbm_text['objective']))

    # Use the same algorithm to parse the tree
    for i, tree in enumerate(gbm_text['tree_info']):
        tree_id = i
        class_id = tree_id % n_classes
        # tree['shrinkage'] --> LightGbm provides figures with it already.
        learning_rate = 1.
        _parse_tree_structure(
            tree_id, class_id, learning_rate, tree['tree_structure'], attrs)

    # Sort nodes_* attributes. For one tree, its node indexes
    # should appear in an ascent order in nodes_nodeids. Nodes
    # from a tree with a smaller tree index should appear
    # before trees with larger indexes in nodes_nodeids.
    node_numbers_per_tree = Counter(attrs['nodes_treeids'])
    tree_number = len(node_numbers_per_tree.keys())
    accumulated_node_numbers = [0] * tree_number
    for i in range(1, tree_number):
        accumulated_node_numbers[i] = (
            accumulated_node_numbers[i - 1] + node_numbers_per_tree[i - 1])
    global_node_indexes = []
    for i in range(len(attrs['nodes_nodeids'])):
        tree_id = attrs['nodes_treeids'][i]
        node_id = attrs['nodes_nodeids'][i]
        global_node_indexes.append(
            accumulated_node_numbers[tree_id] + node_id)
    for k, v in attrs.items():
        if k.startswith('nodes_'):
            merged_indexes = zip(
                copy.deepcopy(global_node_indexes), v)
            sorted_list = [pair[1]
                           for pair in sorted(merged_indexes,
                                              key=lambda x: x[0])]
            attrs[k] = sorted_list

    # Create ONNX object
    if (gbm_text['objective'].startswith('binary') or
            gbm_text['objective'].startswith('multiclass')):
        # Prepare label information for both of TreeEnsembleClassifier
        class_type = onnx_proto.TensorProto.STRING
        if all(isinstance(i, (numbers.Real, bool, np.bool_))
               for i in gbm_model.classes_):
            class_type = onnx_proto.TensorProto.INT64
            class_labels = [int(i) for i in gbm_model.classes_]
            attrs['classlabels_int64s'] = class_labels
        elif all(isinstance(i, str) for i in gbm_model.classes_):
            class_labels = [str(i) for i in gbm_model.classes_]
            attrs['classlabels_strings'] = class_labels
        else:
            raise ValueError(
                'Only string and integer class labels are allowed')

        # Create tree classifier
        probability_tensor_name = scope.get_unique_variable_name(
            'probability_tensor')
        label_tensor_name = scope.get_unique_variable_name('label_tensor')

        container.add_node(
            'TreeEnsembleClassifier', operator.input_full_names,
            [label_tensor_name, probability_tensor_name],
            op_domain='ai.onnx.ml', **attrs)

        prob_tensor = probability_tensor_name

        if gbm_model.boosting_type == 'rf':
            col_index_name = scope.get_unique_variable_name('col_index')
            first_col_name = scope.get_unique_variable_name('first_col')
            zeroth_col_name = scope.get_unique_variable_name('zeroth_col')
            denominator_name = scope.get_unique_variable_name('denominator')
            modified_first_col_name = scope.get_unique_variable_name(
                'modified_first_col')
            unit_float_tensor_name = scope.get_unique_variable_name(
                'unit_float_tensor')
            merged_prob_name = scope.get_unique_variable_name('merged_prob')
            predicted_label_name = scope.get_unique_variable_name(
                'predicted_label')
            classes_name = scope.get_unique_variable_name('classes')
            final_label_name = scope.get_unique_variable_name('final_label')

            container.add_initializer(
                col_index_name, onnx_proto.TensorProto.INT64, [], [1])
            container.add_initializer(
                unit_float_tensor_name, onnx_proto.TensorProto.FLOAT,
                [], [1.0])
            container.add_initializer(
                denominator_name, onnx_proto.TensorProto.FLOAT, [],
                [100.0])
            container.add_initializer(classes_name, class_type,
                                      [len(class_labels)], class_labels)

            container.add_node(
                'ArrayFeatureExtractor',
                [probability_tensor_name, col_index_name],
                first_col_name,
                name=scope.get_unique_operator_name(
                    'ArrayFeatureExtractor'),
                op_domain='ai.onnx.ml')
            apply_div(scope, [first_col_name, denominator_name],
                      modified_first_col_name, container, broadcast=1)
            apply_sub(
                scope, [unit_float_tensor_name, modified_first_col_name],
                zeroth_col_name, container, broadcast=1)
            container.add_node(
                'Concat', [zeroth_col_name, modified_first_col_name],
                merged_prob_name,
                name=scope.get_unique_operator_name('Concat'), axis=1)
            container.add_node(
                'ArgMax', merged_prob_name,
                predicted_label_name,
                name=scope.get_unique_operator_name('ArgMax'), axis=1)
            container.add_node(
                'ArrayFeatureExtractor', [classes_name, predicted_label_name],
                final_label_name,
                name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                op_domain='ai.onnx.ml')
            apply_reshape(scope, final_label_name,
                          operator.outputs[0].full_name,
                          container, desired_shape=[-1, ])
            prob_tensor = merged_prob_name
        else:
            container.add_node('Identity', label_tensor_name,
                               operator.outputs[0].full_name,
                               name=scope.get_unique_operator_name('Identity'))

        # Convert probability tensor to probability map
        # (keys are labels while values are the associated probabilities)
        container.add_node('Identity', prob_tensor,
                           operator.outputs[1].full_name)
    else:
        # Create tree regressor
        output_name = scope.get_unique_variable_name('output')

        keys_to_be_renamed = list(
            k for k in attrs if k.startswith('class_'))

        for k in keys_to_be_renamed:
            # Rename class_* attribute to target_*
            # because TreeEnsebmleClassifier
            # and TreeEnsembleClassifier have different ONNX attributes
            attrs['target' + k[5:]] = copy.deepcopy(attrs[k])
            del attrs[k]
        container.add_node(
            'TreeEnsembleRegressor', operator.input_full_names,
            output_name, op_domain='ai.onnx.ml', **attrs)
        if gbm_model.boosting_type == 'rf':
            denominator_name = scope.get_unique_variable_name('denominator')

            container.add_initializer(
                denominator_name, onnx_proto.TensorProto.FLOAT, [], [100.0])

            apply_div(scope, [output_name, denominator_name],
                      operator.output_full_names, container, broadcast=1)
        else:
            container.add_node('Identity', output_name,
                               operator.output_full_names,
                               name=scope.get_unique_operator_name('Identity'))
        if post_transform:
            _add_post_transform_node(container, post_transform)


def _add_post_transform_node(container: ModelComponentContainer, op_type: str):
    """
    Add a post transform node to a ModelComponentContainer.

    Useful for post transform functions that are not supported by the ONNX spec yet (e.g. 'Exp').
    """
    assert len(container.outputs) == 1, "Adding a post transform node is only possible for models with 1 output."
    original_output_name = container.outputs[0].name
    new_output_name = f"{op_type.lower()}_{original_output_name}"
    post_transform_node = onnx.helper.make_node(op_type, inputs=[original_output_name], outputs=[new_output_name])
    container.nodes.append(post_transform_node)
    container.outputs[0].name = new_output_name


def modify_tree_for_rule_in_set(gbm, use_float=False):
    """
    LightGBM produces sometimes a tree with a node set
    to use rule ``==`` to a set of values (= in set),
    the values are separated by ``||``.
    This function unfold theses nodes.
    """
    if 'tree_info' in gbm:
        for tree in gbm['tree_info']:
            modify_tree_for_rule_in_set(tree, use_float=use_float)
        return

    if 'tree_structure' in gbm:
        modify_tree_for_rule_in_set(gbm['tree_structure'], use_float=use_float)
        return

    if 'decision_type' not in gbm:
        return

    def recursive_call(this):
        if 'left_child' in this:
            modify_tree_for_rule_in_set(
                this['left_child'], use_float=use_float)
        if 'right_child' in this:
            modify_tree_for_rule_in_set(
                this['right_child'], use_float=use_float)

    def str2number(val):
        if use_float:
            return float(val)
        else:
            try:
                return int(val)
            except ValueError:
                return float(val)

    dec = gbm['decision_type']
    if dec != '==':
        return recursive_call(gbm)

    th = gbm['threshold']
    if not isinstance(th, str) or '||' not in th:
        return recursive_call(gbm)

    pos = th.index('||')
    th1 = str2number(th[:pos])

    rest = th[pos + 2:]
    if '||' not in rest:
        rest = str2number(rest)

    gbm['threshold'] = th1
    new_node = gbm.copy()
    gbm['right_child'] = new_node
    new_node['threshold'] = rest
    return recursive_call(gbm)


def convert_lgbm_zipmap(scope, operator, container):
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
    if hasattr(operator, 'classlabels_int64s'):
        zipmap_attrs['classlabels_int64s'] = operator.classlabels_int64s
        to_type = onnx_proto.TensorProto.INT64
    elif hasattr(operator, 'classlabels_strings'):
        zipmap_attrs['classlabels_strings'] = operator.classlabels_strings
        to_type = onnx_proto.TensorProto.STRING
    else:
        raise RuntimeError("Unknown class type.")
    if to_type == onnx_proto.TensorProto.STRING:
        apply_identity(scope, operator.inputs[0].full_name,
                       operator.outputs[0].full_name, container)
    else:
        apply_cast(scope, operator.inputs[0].full_name,
                   operator.outputs[0].full_name, container, to=to_type)

    if operator.zipmap:
        container.add_node('ZipMap', operator.inputs[1].full_name,
                           operator.outputs[1].full_name,
                           op_domain='ai.onnx.ml', **zipmap_attrs)
    else:
        # This should be apply_identity but optimization fails in
        # onnxconverter-common when trying to remove identity nodes.
        apply_clip(scope, operator.inputs[1].full_name,
                   operator.outputs[1].full_name, container,
                   min=np.array([0], dtype=np.float32),
                   max=np.array([1], dtype=np.float32))


register_converter('LgbmClassifier', convert_lightgbm)
register_converter('LgbmRegressor', convert_lightgbm)
register_converter('LgbmZipMap', convert_lgbm_zipmap)

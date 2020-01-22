# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter

_LINK_FUNCTION_TO_POST_TRANSFORM = {
    'identity': 'NONE',
    'logit': 'LOGISTIC',
    'ologit': 'LOGISTIC'
}


def _get_post_transform(params):
    link_function = params["link_function"]
    family = params["family"]
    if family == "multinomial":
        return 'SOFTMAX'
    elif link_function not in _LINK_FUNCTION_TO_POST_TRANSFORM.keys():
        raise ValueError("Link function %s not supported." % link_function)
    else:
        return _LINK_FUNCTION_TO_POST_TRANSFORM[link_function]


def _get_default_tree_attribute_pairs(is_classifier, params):
    attrs = {
        'post_transform': _get_post_transform(params)
    }
    nclasses = params["nclasses"]
    if is_classifier:
        predicted_classes = nclasses if nclasses > 2 else 1
        attrs['base_values'] = [params["base_score"] for _ in range(0, predicted_classes)]
    else:
        attrs['n_targets'] = 1
        attrs['base_values'] = [params["base_score"]]
    for k in {'nodes_treeids',  'nodes_nodeids',
              'nodes_featureids', 'nodes_modes', 'nodes_values',
              'nodes_truenodeids', 'nodes_falsenodeids', 'nodes_missing_value_tracks_true'}:
        attrs[k] = []
    node_attr_prefix = _node_attr_prefix(is_classifier)
    for k in {'_treeids', '_nodeids', '_ids', '_weights'}:
        attrs[node_attr_prefix + k] = []
    return attrs


def _add_node(
        attr_pairs, is_classifier, tree_id, node_id,
        feature_id, mode, value, true_child_id, false_child_id, weights,
        missing
):
    attr_pairs['nodes_treeids'].append(tree_id)
    attr_pairs['nodes_nodeids'].append(node_id)
    attr_pairs['nodes_featureids'].append(feature_id)
    attr_pairs['nodes_modes'].append(mode)
    attr_pairs['nodes_values'].append(float(value))
    attr_pairs['nodes_truenodeids'].append(true_child_id)
    attr_pairs['nodes_falsenodeids'].append(false_child_id)
    attr_pairs['nodes_missing_value_tracks_true'].append(missing)
    if mode == 'LEAF':
        node_attr_prefix = _node_attr_prefix(is_classifier)
        for i, w in enumerate(weights):
            attr_pairs[node_attr_prefix + '_treeids'].append(tree_id)
            attr_pairs[node_attr_prefix + '_nodeids'].append(node_id)
            attr_pairs[node_attr_prefix + '_ids'].append(i)
            attr_pairs[node_attr_prefix + '_weights'].append(float(w))


def _node_attr_prefix(is_classifier):
    return "class" if is_classifier else "target"


def _fill_node_attributes(tree_id, node, attr_pairs, is_classifier):
    if 'leftChild' in node:
        if node["isCategorical"]:
            raise ValueError("categorical splits not supported, use one_hot_explicit")
        else:
            operator = 'BRANCH_GTE'
            value = node['splitValue']
        _add_node(
            attr_pairs=attr_pairs,
            is_classifier=is_classifier,
            tree_id=tree_id,
            mode=operator,
            value=value,
            node_id=node['id'],
            feature_id=node['colId'],
            true_child_id=node['rightChild']['id'],
            false_child_id=node['leftChild']['id'],
            weights=None,
            missing=(0 if node["leftward"] else 1),
        )
        _fill_node_attributes(tree_id, node["leftChild"], attr_pairs, is_classifier)
        _fill_node_attributes(tree_id, node["rightChild"], attr_pairs, is_classifier)
    else:  # leaf
        weights = [node['predValue']]
        _add_node(
            attr_pairs=attr_pairs,
            is_classifier=is_classifier,
            tree_id=tree_id,
            value=0.,
            node_id=node['id'],
            feature_id=0, mode='LEAF',
            true_child_id=0, false_child_id=0,
            weights=weights,
            missing=False
        )


def assign_node_ids(node, next_id):
    if node is None:
        return next_id
    node["id"] = next_id
    next_id += 1
    next_id = assign_node_ids(node.get("leftChild", None), next_id)
    return assign_node_ids(node.get("rightChild", None), next_id)


def fill_tree_attributes(model, attr_pairs, node_attr_prefix):
    for tree in model["trees"]:
        assign_node_ids(tree["root"], 0)
        _fill_node_attributes(tree["index"], tree["root"], attr_pairs, node_attr_prefix)


def convert_regression(scope, operator, container, params):
    model = operator.raw_operator

    attr_pairs = _get_default_tree_attribute_pairs(False, params)
    fill_tree_attributes(model, attr_pairs, False)

    # add nodes
    container.add_node('TreeEnsembleRegressor', operator.input_full_names,
                       operator.output_full_names, op_domain='ai.onnx.ml',
                       name=scope.get_unique_operator_name('TreeEnsembleRegressor'), **attr_pairs)


def convert_classifier(scope, operator, container, params):
    if params["family"] == "multinomial" and params["nclasses"] == 2:
        raise ValueError("Multinomial distribution with two classes not supported, use binomial distribution.")
    model = operator.raw_operator

    attr_pairs = _get_default_tree_attribute_pairs(True, params)
    fill_tree_attributes(model, attr_pairs, True)

    n_trees_in_group = params["n_trees_in_group"]
    attr_pairs['class_ids'] = [v % n_trees_in_group for v in attr_pairs['class_treeids']]
    attr_pairs['classlabels_strings'] = params["class_labels"]

    container.add_node('TreeEnsembleClassifier', operator.input_full_names,
                       operator.output_full_names,
                       op_domain='ai.onnx.ml',
                       name=scope.get_unique_operator_name('TreeEnsembleClassifier'),
                       **attr_pairs)


def convert_h2o(scope, operator, container):
    params = operator.raw_operator["params"]
    is_classifier = params["classifier"]
    if is_classifier:
        convert_classifier(scope, operator, container, params)
    else:
        convert_regression(scope, operator, container, params)


register_converter('H2OTreeMojo', convert_h2o)

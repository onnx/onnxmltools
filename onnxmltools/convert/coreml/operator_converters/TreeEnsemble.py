# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter

COREML_TREE_NODE_BEHAVIOR_TO_ONNX_TREE_NODE_MODE = {
    0: 'BRANCH_LEQ',
    1: 'BRANCH_LT',
    2: 'BRANCH_GTE',
    3: 'BRANCH_GT',
    4: 'BRANCH_EQ',
    5: 'BRANCH_NEQ',
    6: 'LEAF'
}

COREML_TREE_POST_TRANSFORM_TO_ONNX_TREE_POST_TRANSFORM = {
    0: 'NONE',
    1: 'SOFTMAX',
    2: 'LOGISTIC',
    3: 'SOFTMAX_ZERO'
}


def get_onnx_tree_mode(cm_tree_behavior):
    if cm_tree_behavior in COREML_TREE_NODE_BEHAVIOR_TO_ONNX_TREE_NODE_MODE:
        return COREML_TREE_NODE_BEHAVIOR_TO_ONNX_TREE_NODE_MODE[cm_tree_behavior]
    raise ValueError('CoreML tree node behavior not supported {0}'.format(cm_tree_behavior))


def get_onnx_tree_post_transform(cm_tree_post_transform):
    if cm_tree_post_transform in COREML_TREE_POST_TRANSFORM_TO_ONNX_TREE_POST_TRANSFORM:
        return COREML_TREE_POST_TRANSFORM_TO_ONNX_TREE_POST_TRANSFORM[cm_tree_post_transform]
    raise ValueError('CoreML tree post transform not supported {0}'.format(cm_tree_post_transform))


def convert_tree_ensemble_model(scope, operator, container):
    raw_model = operator.raw_operator
    attrs = {'name': operator.full_name}
    if raw_model.WhichOneof('Type') == 'treeEnsembleClassifier':
        op_type = 'TreeEnsembleClassifier'
        prefix = 'class'
        nodes = raw_model.treeEnsembleClassifier.treeEnsemble.nodes
        attrs['base_values'] = raw_model.treeEnsembleClassifier.treeEnsemble.basePredictionValue
        attrs['post_transform'] = get_onnx_tree_post_transform(raw_model.treeEnsembleClassifier.postEvaluationTransform)
        zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
        if raw_model.treeEnsembleClassifier.WhichOneof('ClassLabels') == 'int64ClassLabels':
            class_labels = list(int(i) for i in raw_model.treeEnsembleClassifier.int64ClassLabels.vector)
            attrs['classlabels_int64s'] = class_labels
            zipmap_attrs['classlabels_int64s'] = class_labels
        else:
            class_labels = list(s.encode('utf-8') for s in raw_model.treeEnsembleClassifier.stringClassLabels.vector)
            attrs['classlabels_strings'] = class_labels
            zipmap_attrs['classlabels_strings'] = class_labels
    elif raw_model.WhichOneof('Type') == 'treeEnsembleRegressor':
        op_type = 'TreeEnsembleRegressor'
        prefix = 'target'
        nodes = raw_model.treeEnsembleRegressor.treeEnsemble.nodes
        attrs['base_values'] = raw_model.treeEnsembleRegressor.treeEnsemble.basePredictionValue
        attrs['n_targets'] = raw_model.treeEnsembleRegressor.treeEnsemble.numPredictionDimensions
        attrs['post_transform'] = get_onnx_tree_post_transform(raw_model.treeEnsembleRegressor.postEvaluationTransform)
    else:
        raise ValueError('Unknown tree model type')

    leaf_treeids = [node.treeId for node in nodes if 6 == node.nodeBehavior for _ in node.evaluationInfo]
    leaf_nodeids = [node.nodeId for node in nodes if 6 == node.nodeBehavior for _ in node.evaluationInfo]
    leaf_ids = [weight.evaluationIndex for node in nodes if 6 == node.nodeBehavior for weight in node.evaluationInfo]

    leaf_weights = [weight.evaluationValue for node in nodes if 6 == node.nodeBehavior for weight in
                    node.evaluationInfo]

    assert (len(leaf_ids) == len(leaf_weights))
    assert (len(leaf_weights) == len(leaf_nodeids))
    assert (len(leaf_nodeids) == len(leaf_treeids))

    nodes_nodeids = [x.nodeId for x in nodes]
    nodes_treeids = [x.treeId for x in nodes]
    nodes_featureids = [x.branchFeatureIndex for x in nodes]
    nodes_values = [x.branchFeatureValue for x in nodes]
    nodes_truenodeids = [x.trueChildNodeId for x in nodes]
    nodes_falsenodeids = [x.falseChildNodeId for x in nodes]
    nodes_missing_value_tracks_true = [x.missingValueTracksTrueChild for x in nodes]
    nodes_hitrates = [float(x.relativeHitRate) for x in nodes]
    nodes_modes = [get_onnx_tree_mode(x.nodeBehavior) for x in nodes]

    attrs['nodes_treeids'] = nodes_treeids
    attrs['nodes_nodeids'] = nodes_nodeids
    attrs['nodes_featureids'] = nodes_featureids
    attrs['nodes_values'] = nodes_values
    attrs['nodes_hitrates'] = nodes_hitrates
    attrs['nodes_modes'] = nodes_modes
    attrs['nodes_truenodeids'] = nodes_truenodeids
    attrs['nodes_falsenodeids'] = nodes_falsenodeids
    attrs['nodes_missing_value_tracks_true'] = nodes_missing_value_tracks_true
    attrs[prefix + '_treeids'] = leaf_treeids
    attrs[prefix + '_nodeids'] = leaf_nodeids
    attrs[prefix + '_ids'] = leaf_ids
    attrs[prefix + '_weights'] = leaf_weights

    # For regression, we can simply construct a model. For classifier, due to the different representation of
    # classes' probabilities, we need to add some operators for type conversion.
    if raw_model.WhichOneof('Type') == 'treeEnsembleRegressor':
        # Create ONNX representation of this operator. If there is only one input, its full topology is
        #
        # input features ---> TreeEnsembleRegressor ---> output
        #
        # If there are multiple (e.g., "N" features) input features, we need to concatenate them all together before feeding them into
        # ONNX tree-based model. It leads to the following computational graph.
        #
        # input feature 1 -----.
        #        ...           |
        #        ...           v
        #        ...      ---> Feature Vectorizer ---> TreeEnsembleRegressor ---> output
        #        ...           ^
        #        ...           |
        # input feature N -----'
        if len(operator.inputs) > 1:
            feature_vector_name = scope.get_unique_variable_name('feature_vector')
            container.add_node('FeatureVectorizer', operator.input_full_names, feature_vector_name,
                               op_domain='ai.onnx.ml', name=scope.get_unique_operator_name('FeatureVectorizer'),
                               inputdimensions=[variable.type.shape[1] for variable in operator.inputs])
            container.add_node(op_type, feature_vector_name, operator.output_full_names,
                               op_domain='ai.onnx.ml', **attrs)
        else:
            container.add_node(op_type, operator.input_full_names, operator.output_full_names,
                               op_domain='ai.onnx.ml', **attrs)
    else:
        # For classifiers, due to the different representation of classes' probabilities, we need to add some
        # operators for type conversion. It turns out that we have the following topology.
        # input features ---> TreeEnsembleClassifier ---> label (must present)
        #                               |
        #                               '--> probability tensor ---> ZipMap ---> probability map (optional)
        #
        # Similar to the regressor's case, if there are multiple input features, we need to concatenate them all
        # together before feeding them into ONNX tree-based model. It leads to the following computational graph.
        #
        # input feature 1 -----.
        #        ...           |
        #        ...           v
        #        ...      ---> Feature Vectorizer ---> TreeEnsembleClassifier ---> label (must present)
        #        ...           ^                                 |
        #        ...           |                                 '--> probability tensor ---> ZipMap ---> probability
        # input feature N -----'                                                                          map (optional)

        # Set up input feature(s)
        if len(operator.inputs) > 1:
            feature_vector_name = scope.get_unique_variable_name('feature_vector')
            container.add_node('FeatureVectorizer', operator.input_full_names, feature_vector_name,
                               op_domain='ai.onnx.ml', name=scope.get_unique_operator_name('FeatureVectorizer'),
                               inputdimensions=[variable.type.shape[1] for variable in operator.inputs])
        else:
            feature_vector_name = operator.inputs[0].full_name

        # Find label name and probability name
        proba_output_name = None
        for variable in operator.outputs:
            if raw_model.description.predictedFeatureName == variable.raw_name:
                label_output_name = variable.full_name
            if raw_model.description.predictedProbabilitiesName != '' and raw_model.description.predictedProbabilitiesName == variable.raw_name:
                proba_output_name = variable.full_name

        proba_tensor_name = scope.get_unique_variable_name('ProbabilityTensor')

        if proba_output_name is not None:
            # Add tree model ONNX node with probability output
            container.add_node(op_type, feature_vector_name, [label_output_name, proba_tensor_name],
                               op_domain='ai.onnx.ml', **attrs)

            # Add ZipMap to convert probability tensor into probability map
            container.add_node('ZipMap', [proba_tensor_name], [proba_output_name],
                               op_domain='ai.onnx.ml', **zipmap_attrs)
        else:
            # Add support vector classifier without probability output
            container.add_node(op_type, feature_vector_name, [label_output_name, proba_tensor_name],
                               op_domain='ai.onnx.ml', **attrs)


register_converter("treeEnsembleClassifier", convert_tree_ensemble_model)
register_converter("treeEnsembleRegressor", convert_tree_ensemble_model)

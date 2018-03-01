#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import NodeBuilder, model_util
from .datatype import get_onnx_tree_mode
from .datatype import get_onnx_tree_post_transform

# This function is used to convert CoreML tree-based model into a ONNX representation which can include.
# multiple operators.
#
# The conversion of tree-based regressor is straightforward; the topology of CoreML and its ONNX counterpart are the
# same. See the following two visualization.
#
# Symbols:
#  X: input feature vector
#  Y: target vector
#  P: probability dicionary. It only appears in a classifer. A key-value pair (k, v) means the probability of class "k"
#     is "v."
#  T: probability tensor.
#
# CoreML computational graph for tree-based regressor:
#  X ---> CoreML Tree-based regressor ---> Y
#
# ONNX computational graph for tree-based regressor:
#  X ---> ONNX Tree-based regressor ---> Y
#
# To convert CoreML's tree-based classifiers, we sometime may add some extra operators because class probabilities are
# stored in a tensor in ONNX but a dictionary in CoreML. See the subsequent two graphs.
#
# CoreML computational graph for tree-based classifier:
#  X ---> CoreML Tree-based regressor ---> Y
#                    |
#                    '---> P (optional)
#
# ONNX computational graph for tree-based classifier:
#  X ---> ONNX Tree-based regressor ---> Y
#                    |
#                    '---> T ---> ZipMap ---> P (things after and including ZipMap are optional. If the original CoreML
#                                                model has no "P," the corresponding ONNX model doesn't have ZipMap and
#                                                "P.")
def convert(context, cm_node, inputs, outputs, prefix):
    zipmap_node = None
    if prefix == "class":
        # Set up things for tree-based classifiers
        nb = NodeBuilder(context, 'TreeEnsembleClassifier', op_domain='ai.onnx.ml')
        nodes = cm_node.treeEnsembleClassifier.treeEnsemble.nodes
        nb.add_attribute('base_values', cm_node.treeEnsembleClassifier.treeEnsemble.basePredictionValue)
        post_transform = get_onnx_tree_post_transform(cm_node.treeEnsembleClassifier.postEvaluationTransform)

        # Assign class labels to ONNX model
        if cm_node.treeEnsembleClassifier.WhichOneof('ClassLabels') == 'stringClassLabels':
            class_labels = list(str(s) for s in cm_node.treeEnsembleClassifier.stringClassLabels.vector)
            nb.add_attribute('classlabels_strings', class_labels)
        elif cm_node.treeEnsembleClassifier.WhichOneof('ClassLabels') == 'int64ClassLabels':
            class_labels = list(int(i) for i in cm_node.treeEnsembleClassifier.int64ClassLabels.vector)
            nb.add_attribute('classlabels_int64s', class_labels)
        else:
            raise ValueError('Unknown class label type')

        # Find the ONNX name for the predicted label in CoreML
        predicted_label_name = context.get_onnx_name(cm_node.description.predictedFeatureName)
        nb.add_output(predicted_label_name)

        # Create variable name to store the class probabilities produced by ONNX tree-based classifier
        probability_tensor_name = context.get_unique_name('probability_tensor')
        nb.add_output(probability_tensor_name)

        # Class probabilities are encoded by a tensor in ONNX but a dictionary in CoreML. We therefore allocate a ZipMap
        # operator to convert the probability tensor produced by ONNX's tree-based classifier into a dictionary.
        if cm_node.description.predictedProbabilitiesName != '':
            # Find the corresponding ONNX name for CoreML's probability output (a dictionary)
            predicted_probability_name = context.get_onnx_name(cm_node.description.predictedProbabilitiesName)
            # Create a ZipMap to connect probability tensor and probability dictionary
            zipmap_node = model_util.make_zipmap_node(context, probability_tensor_name,
                                                      predicted_probability_name, class_labels)
    elif prefix == "target":
        # Set up things for tree-based regressors
        nb = NodeBuilder(context, 'TreeEnsembleRegressor', op_domain='ai.onnx.ml')
        nb.extend_outputs(outputs)
        nodes = cm_node.treeEnsembleRegressor.treeEnsemble.nodes
        nb.add_attribute('base_values', cm_node.treeEnsembleRegressor.treeEnsemble.basePredictionValue)
        post_transform = get_onnx_tree_post_transform(cm_node.treeEnsembleRegressor.postEvaluationTransform)
    else:
        raise TypeError("Unknown tree type: prefix='{0}'".format(prefix))

    leaf_treeids = [node.treeId for node in nodes if 6 == node.nodeBehavior for weight in node.evaluationInfo]
    leaf_nodeids = [node.nodeId for node in nodes if 6 == node.nodeBehavior for weight in node.evaluationInfo]
    leaf_ids = [weight.evaluationIndex for node in nodes if 6 == node.nodeBehavior for weight in node.evaluationInfo]
    leaf_weights = [weight.evaluationValue for node in nodes if 6 == node.nodeBehavior
                    for weight in node.evaluationInfo]

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

    nb.add_attribute('nodes_treeids', nodes_treeids)
    nb.add_attribute('nodes_nodeids', nodes_nodeids)
    nb.add_attribute('nodes_featureids', nodes_featureids)
    nb.add_attribute('nodes_values', nodes_values)
    nb.add_attribute('nodes_hitrates', nodes_hitrates)
    nb.add_attribute('nodes_modes', nodes_modes)
    nb.add_attribute('nodes_truenodeids', nodes_truenodeids)
    nb.add_attribute('nodes_falsenodeids', nodes_falsenodeids)
    nb.add_attribute('nodes_missing_value_tracks_true', nodes_missing_value_tracks_true)
    nb.add_attribute('post_transform', post_transform)
    nb.add_attribute(prefix + '_treeids', leaf_treeids)
    nb.add_attribute(prefix + '_nodeids', leaf_nodeids)
    nb.add_attribute(prefix + '_ids', leaf_ids)
    nb.add_attribute(prefix + '_weights', leaf_weights)
    nb.extend_inputs(inputs)

    if zipmap_node is None:
        return [nb.make_node()]
    else:
        # Notice that ZipMap must come after the tree model because ONNX nodes in a graph must be sorted according to
        # the evaluation order.
        return [nb.make_node(), zipmap_node]

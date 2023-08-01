# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ...common._registration import register_converter


def convert_glm_classifier(scope, operator, container):
    # For classifiers, due to the different representations of classes' probabilities in ONNX and CoreML, some extra
    # operators are required. See explanation below.
    #
    # Symbols:
    #  X: input feature vector
    #  Y: the best output label (i.e., the one with highest probability)
    #  P: probability dictionary of all classes. Its keys are class labels and its values are those labels'
    #     probabilities.
    #
    #  T: probability tensor produced by ONNX classifier
    #  T': normalized version of "T." Its sum must be 1.
    #
    # CoreML computational graph (binary class and multi-class classifications):
    #
    #            X ---> CoreML GLMClassifier ---> Y (must present in model)
    #                           |
    #                           '---------------> P
    #
    # ONNX computational graph (binary class classification):
    #
    #            X ---> ONNX GLMClassifier ---> Y (must present in model)
    #                           |
    #                           '-------------> T ---> ZipMap ---> P (If P is not specified in the considered CoreML
    #                                                                 model "T" would become an isolated variable
    #                                                                 which is not connected with any other
    #                                                                 operators. That is, both of ZipMap and P are
    #                                                                 optional in this graph.)
    #
    # ONNX computational graph (multi-class classification):
    #
    #            X ---> ONNX GLMClassifier ---> Y (must present in model)
    #                           |
    #                           '-------------> T ---> L1-norm Normalizer ---> T' ---> ZipMap ---> P
    #                                                                              (things after T' are optional.
    #                                                                               If P is specified, we may have
    #                                                                               ZipMap and P. Otherwise, this
    #                                                                               ends at T' and T' is not linked
    #                                                                               with other operators.
    from coremltools.proto.GLMClassifier_pb2 import GLMClassifier
    op_type = 'LinearClassifier'
    attrs = {'name': operator.full_name}
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
    glm = operator.raw_operator.glmClassifier

    transform_table = {GLMClassifier.Logit: 'LOGISTIC', GLMClassifier.Probit: 'PROBIT'}
    if glm.postEvaluationTransform not in transform_table:
        raise ValueError('Unsupported post-transformation: {}'.format(glm.postEvaluationTransform))
    attrs['post_transform'] = transform_table[glm.postEvaluationTransform]

    encoding_table = {GLMClassifier.ReferenceClass: True, GLMClassifier.OneVsRest: False}
    if glm.classEncoding not in encoding_table:
        raise ValueError('Unsupported class encoding: {}'.format(glm.classEncoding))
    attrs['multi_class'] = encoding_table[glm.classEncoding]

    # Determine the dimensionality of the model weights.
    dim_target = len(glm.weights)
    dim_feature = len(glm.weights[0].value)

    matrix_w = np.ndarray(shape=(dim_target, dim_feature))
    for i, w in enumerate(glm.weights):
        matrix_w[i, :] = w.value

    if glm.WhichOneof('ClassLabels') == 'stringClassLabels':
        class_labels = list(s.encode('utf-8') for s in glm.stringClassLabels.vector)
        attrs['classlabels_strings'] = class_labels
        zipmap_attrs['classlabels_strings'] = class_labels
    elif glm.WhichOneof('ClassLabels') == 'int64ClassLabels':
        class_labels = list(int(i) for i in glm.int64ClassLabels.vector)
        attrs['classlabels_ints'] = class_labels
        zipmap_attrs['classlabels_int64s'] = class_labels
    else:
        raise ValueError('Unknown class label type')

    coefficients = matrix_w.flatten().tolist()
    intercepts = list(float(i) for i in glm.offset)
    if len(class_labels) == 2:
        # Handle the binary case for coefficients and intercepts
        coefficients = list(map(lambda x: -1 * x, coefficients)) + coefficients
        intercepts = list(map(lambda x: -1 * x, intercepts)) + intercepts

    attrs['coefficients'] = coefficients
    attrs['intercepts'] = intercepts

    # For classifiers, due to the different representation of classes' probabilities, we need to add some
    # operators for type conversion. It turns out that we have the following topology.
    # input features ---> GLMClassifier ---> label (must present)
    #                           |
    #                           '--> probability tensor ---> Normalizer ---> normalized ---> ZipMap ---> probability map
    #                                                (depending on whether probability output exists in CoreML model,
    #                                                 variables/operators after probability tensor may disappear)
    raw_model = operator.raw_operator
    # Find label name and probability name
    proba_output_name = None
    for variable in operator.outputs:
        if raw_model.description.predictedFeatureName == variable.raw_name:
            label_output_name = variable.full_name
        if raw_model.description.predictedProbabilitiesName != '' and \
                raw_model.description.predictedProbabilitiesName == variable.raw_name:
            proba_output_name = variable.full_name

    inputs = [variable.full_name for variable in operator.inputs]
    proba_tensor_name = scope.get_unique_variable_name('ProbabilityTensor')

    if proba_output_name is not None:
        # Add tree model ONNX node with probability output
        container.add_node(op_type, inputs, [label_output_name, proba_tensor_name], op_domain='ai.onnx.ml', **attrs)

        # Add a normalizer to make sure that the sum of all classes' probabilities is 1. It doesn't affect binary
        # classification. For multi-class clssifiers, if one applies sigmoid function independently to all raw scores,
        # we have to add a normalization so that the sum of all probabilities remains 1. Of course, if softmax is used
        # to convert raw scores into probabilities, this normalization doesn't change anything.
        if len(class_labels) > 2:
            normalized_proba_tensor_name = scope.get_unique_variable_name(proba_tensor_name + '_normalized')
            container.add_node('Normalizer', proba_tensor_name, normalized_proba_tensor_name, op_domain='ai.onnx.ml',
                               name=scope.get_unique_operator_name('Normalizer'), norm='L1')
        else:
            # If we don't need a normalization, we just pass the original probability tensor to the following ZipMap
            normalized_proba_tensor_name = proba_tensor_name

        # Add ZipMap to convert normalized probability tensor into probability map
        container.add_node('ZipMap', [normalized_proba_tensor_name], [proba_output_name],
                           op_domain='ai.onnx.ml', **zipmap_attrs)
    else:
        # Add linear classifier with isolated probability output, which means that the probability
        # tensor won't be accessed by any others.
        container.add_node(op_type, inputs, [label_output_name, proba_tensor_name], op_domain='ai.onnx.ml', **attrs)


register_converter('glmClassifier', convert_glm_classifier)

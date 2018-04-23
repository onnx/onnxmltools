# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import itertools
import numpy as np
from ...common._registration import register_converter


def extract_support_vectors_as_dense_tensor(svm_model):
    support_type = svm_model.WhichOneof('supportVectors')
    if support_type == 'denseSupportVectors':
        vectors = svm_model.denseSupportVectors.vectors
        support_vectors = np.array([v.values for v in vectors]).flatten()
    elif support_type == 'sparseSupportVectors':
        # Since current ONNX doesn't support sparse representations, we have to save them as dense tensors. It may
        # dramatically prolong prediction time and increase memory usage.
        vectors = svm_model.sparseSupportVectors.vectors
        # Search for the maximum dimension over all sparse support vectors
        max_idx = 0
        for v in vectors:
            for n in v.nodes:
                if n.index > max_idx:
                    max_idx = n.index
        # Save sparse vectors to dense vectors
        support_vectors = np.zeros(shape=(len(vectors), max_idx))
        for i, v in enumerate(vectors):
            for n in v.nodes:
                support_vectors[i][n.index - 1] = n.value
        support_vectors = support_vectors.flatten()
    else:
        raise ValueError('Unsupported support vector type: %s' % support_type)
    return len(vectors), support_vectors


def convert_svm_classifier(scope, operator, container):
    params = operator.raw_operator.supportVectorClassifier
    kernel_enum = {'linearKernel': 'LINEAR', 'polyKernel': 'POLY',
                   'rbfKernel': 'RBF', 'sigmoidKernel': 'SIGMOID', 'precomputedKernel': 'PRECOMPUTED'}
    kernel = params.kernel
    kernel_val = kernel.WhichOneof('kernel')
    svc_kernel = kernel_enum[kernel_val]

    if kernel_val == 'rbfKernel':
        svc_kernel_params = [kernel.rbfKernel.gamma, 0.0, 0.0]
    elif kernel_val == 'polyKernel':
        svc_kernel_params = [kernel.polyKernel.gamma,
                             kernel.polyKernel.coef0, kernel.polyKernel.degree]
    elif kernel_val == 'sigmoidKernel':
        svc_kernel_params = [kernel.sigmoidKernel.gamma,
                             kernel.sigmoidKernel.coef0, 0.0]
    elif kernel_val == 'linearKernel':
        svc_kernel_params = [0.0, 0.0, 0.0]

    prob_a = params.probA
    prob_b = params.probB
    support_vectors_per_class = params.numberOfSupportVectorsPerClass
    n_supports, svc_support_vectors = extract_support_vectors_as_dense_tensor(
        operator.raw_operator.supportVectorClassifier)
    chain_coef = list(itertools.chain.from_iterable([coef.alpha for coef in params.coefficients]))
    svc_coefficients = chain_coef
    svc_rho = [-x for x in params.rho]

    op_type = 'SVMClassifier'
    op_name = scope.get_unique_operator_name(op_type)
    attrs = {'name': op_name}
    attrs['kernel_type'] = svc_kernel
    attrs['kernel_params'] = svc_kernel_params
    if prob_a:
        attrs['prob_a'] = prob_a
    if prob_b:
        attrs['prob_b'] = prob_b
    attrs['vectors_per_class'] = support_vectors_per_class
    attrs['support_vectors'] = svc_support_vectors
    attrs['coefficients'] = svc_coefficients
    attrs['rho'] = svc_rho
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
    svc_classes = params.WhichOneof('ClassLabels')
    if svc_classes == 'int64ClassLabels':
        class_labels = list(int(i) for i in params.int64ClassLabels.vector)
        attrs['classlabels_ints'] = class_labels
        zipmap_attrs['classlabels_int64s'] = class_labels
    elif svc_classes == 'stringClassLabels':
        class_labels = list(str(s).encode('ascii') for s in params.stringClassLabels.vector)
        attrs['classlabels_strings'] = class_labels
        zipmap_attrs['classlabels_strings'] = class_labels
    else:
        raise ValueError('Unknown class label type')

    # For classifiers, due to the different representation of classes' probabilities, we need to add some
    # operators for type conversion. It turns out that we have the following topology.
    # input features ---> SupportVectorClassifier ---> label (must present)
    #                               |
    #                               '--> probability tensor ---> ZipMap ---> probability map (optional)

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
        # Add support vector classifier in terms of ONNX node with probability output
        container.add_node(op_type, inputs, [label_output_name, proba_tensor_name], op_domain='ai.onnx.ml', **attrs)

        # Add ZipMap to convert probability tensor into probability map
        container.add_node('ZipMap', [proba_tensor_name], [proba_output_name],
                           op_domain='ai.onnx.ml', **zipmap_attrs)
    else:
        # Add support vector classifier in terms of ONNX node
        container.add_node(op_type, inputs, [label_output_name, proba_tensor_name], op_domain='ai.onnx.ml', **attrs)


register_converter('supportVectorClassifier', convert_svm_classifier)

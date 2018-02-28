#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
import itertools
from ..common import NodeBuilder
from ..common import register_converter
from ..common import utils
from ..common import model_util


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
                support_vectors[i][n.index-1] = n.value
        support_vectors = support_vectors.flatten()
    else:
        raise ValueError('Unsupported support vector type: %s' % support_type)
    return len(vectors), support_vectors

class SupportVectorClassifierConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'supportVectorClassifier')
            utils._check_has_attr(cm_node.supportVectorClassifier, 'kernel')
            utils._check_has_attr(cm_node.supportVectorClassifier, 'numberOfSupportVectorsPerClass')
            utils._check_has_attr(cm_node.supportVectorClassifier, 'coefficients')
            utils._check_has_attr(cm_node.supportVectorClassifier.coefficients[0], 'alpha')
            utils._check_has_attr(cm_node.supportVectorClassifier, 'rho')
        except AttributeError as e:
            raise RuntimeError("Missing type from CoreML node:" + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """Converts a CoreML SVC to ONNX"""
        kernel_enum = {'linearKernel': 'LINEAR', 'polyKernel': 'POLY',
                       'rbfKernel': 'RBF', 'sigmoidKernel': 'SIGMOID', 'precomputedKernel': 'PRECOMPUTED'}
        kernel = cm_node.supportVectorClassifier.kernel
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

        prob_a = cm_node.supportVectorClassifier.probA
        prob_b = cm_node.supportVectorClassifier.probB
        svc_vectors_per_class = cm_node.supportVectorClassifier.numberOfSupportVectorsPerClass
        n_supports, svc_support_vectors = extract_support_vectors_as_dense_tensor(cm_node.supportVectorClassifier)
        chain_coef = list(itertools.chain.from_iterable(
            [coef.alpha for coef in cm_node.supportVectorClassifier.coefficients]))
        svc_coefficients = chain_coef
        svc_rho = [-x for x in cm_node.supportVectorClassifier.rho]

        nb = NodeBuilder(context, 'SVMClassifier', op_domain='ai.onnx.ml')
        nb.add_attribute('kernel_type', svc_kernel)
        nb.add_attribute('kernel_params', svc_kernel_params)
        if prob_a:
            nb.add_attribute('prob_a', prob_a)
        if prob_b:
            nb.add_attribute('prob_b', prob_b)
        nb.add_attribute('vectors_per_class', svc_vectors_per_class)
        nb.add_attribute('support_vectors', svc_support_vectors)
        nb.add_attribute('coefficients', svc_coefficients)
        nb.add_attribute('rho', svc_rho)
        svc_classes = cm_node.supportVectorClassifier.WhichOneof('ClassLabels')
        if svc_classes == 'int64ClassLabels':
            class_labels = list(int(i) for i in cm_node.supportVectorClassifier.int64ClassLabels.vector)
            nb.add_attribute('classlabels_ints', class_labels)
        elif svc_classes == 'stringClassLabels':
            class_labels = list(str(s) for s in cm_node.supportVectorClassifier.stringClassLabels.vector)
            nb.add_attribute('classlabels_strings', class_labels)

        nb.extend_inputs(inputs)

        # Find the ONNX name for the predicted label in CoreML
        predicted_label_name = context.get_onnx_name(cm_node.description.predictedFeatureName)
        nb.add_output(predicted_label_name)

        # The variable used to store the class probabilities produced by ONNX linear classifier
        probability_tensor_name = context.get_unique_name('probability_tensor')
        nb.add_output(probability_tensor_name)

        nodes = [nb.make_node()]

        if cm_node.description.predictedProbabilitiesName != '':
            # Find the corresponding ONNX name for CoreML's probability output (a dictionary)
            predicted_probability_name = context.get_onnx_name(cm_node.description.predictedProbabilitiesName)
            # Create a ZipMap to connect probability tensor and probability dictionary
            nodes.append(model_util.make_zipmap_node(context, probability_tensor_name,
                                                     predicted_probability_name, class_labels))

        return nodes

# Register the class for processing
register_converter("supportVectorClassifier", SupportVectorClassifierConverter)

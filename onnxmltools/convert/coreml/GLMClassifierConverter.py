#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy
from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from ..common import model_util
from coremltools.proto.GLMClassifier_pb2 import GLMClassifier


class GLMClassifierConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'glmClassifier')
            utils._check_has_attr(cm_node.glmClassifier, 'weights')
            utils._check_has_attr(cm_node.glmClassifier, 'offset')
            utils._check_has_attr(cm_node.glmClassifier, 'classEncoding')
            utils._check_has_attr(cm_node.glmClassifier,
                                  'postEvaluationTransform')
        except AttributeError as e:
            raise RuntimeError("Missing type from CoreML node:" + str(e))

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
    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        glm = cm_node.glmClassifier
        nb = NodeBuilder(context, 'LinearClassifier')
        nb.extend_inputs(inputs)

        transform_table = {GLMClassifier.Logit: 'LOGISTIC', GLMClassifier.Probit: 'PROBIT'}
        if glm.postEvaluationTransform not in transform_table:
            raise ValueError('Unsupported post-transformation: {}'.format(glm.postEvaluationTransform))
        nb.add_attribute('post_transform', transform_table[glm.postEvaluationTransform])

        encoding_table = {GLMClassifier.ReferenceClass: True,
                          GLMClassifier.OneVsRest: False}
        if glm.classEncoding not in encoding_table:
            raise ValueError('Unsupported class encoding: {}'.format(glm.classEncoding))
        nb.add_attribute('multi_class', encoding_table[glm.classEncoding])

        # Determine the dimensionality of the model weights.
        dim_target = len(glm.weights)
        dim_feature = len(glm.weights[0].value)

        matrix_w = numpy.ndarray(shape=(dim_target, dim_feature))
        for i, w in enumerate(glm.weights):
            matrix_w[i, :] = w.value

        if glm.WhichOneof('ClassLabels') == 'stringClassLabels':
            class_labels = list(str(s) for s in glm.stringClassLabels.vector)
            nb.add_attribute('classlabels_strings', class_labels)
        elif glm.WhichOneof('ClassLabels') == 'int64ClassLabels':
            class_labels = list(int(i) for i in glm.int64ClassLabels.vector)
            nb.add_attribute('classlabels_ints', class_labels)
        else:
            raise ValueError('Unknown class label type')

        coefficients = matrix_w.flatten().tolist()
        intercepts = utils.cast_list(float, glm.offset)
        if len(class_labels) == 2:
            # If it's binary classification, we duplicate the coefficients and the intercept for the negative class.
            # For each of the following two variables, the first 50% elements are for the class indexed by 0 while the
            # second 50% are for the class indexed by 1.
            coefficients = list(map(lambda x: -1 * x, coefficients)) + coefficients
            intercepts = list(map(lambda x: -1 * x, intercepts)) + intercepts

        nb.add_attribute('coefficients', coefficients)
        nb.add_attribute('intercepts', intercepts)

        # Find the ONNX name for the predicted label in CoreML
        predicted_label_name = context.get_onnx_name(cm_node.description.predictedFeatureName)
        nb.add_output(predicted_label_name)

        # The variable used to store the class probabilities produced by ONNX linear classifier
        probability_tensor_name = context.get_unique_name('probability_tensor')
        nb.add_output(probability_tensor_name)

        # If it's multi-class classification, we normalize the probability tensor to make sure that the sum of all
        # probabilities is 1
        normalizer_builder = None
        if len(class_labels) > 2 and glm.postEvaluationTransform == GLMClassifier.Logit:
            normalizer_builder = NodeBuilder(context, 'Normalizer')
            normalizer_builder.add_attribute('norm', 'L1')
            normalizer_builder.add_input(probability_tensor_name)
            # Change the name stored in probability_tensor_name so that subsequent operators can access the fixed tensor
            probability_tensor_name = context.get_unique_name('normalized_probability_tensor')
            normalizer_builder.add_output(probability_tensor_name)

        nodes = [builder.make_node() for builder in [nb, normalizer_builder] if builder is not None]

        # Class probabilities are encoded by a tensor in ONNX but a dictionary in CoreML. We therefore allocate a ZipMap
        # operator to convert the probability tensor produced by ONNX's generalized lienar classifier into a dictionary
        # CoreML wants.
        if cm_node.description.predictedProbabilitiesName != '':
            # Find the corresponding ONNX name for CoreML's probability output (a dictionary)
            predicted_probability_name = context.get_onnx_name(cm_node.description.predictedProbabilitiesName)
            # Create a ZipMap to connect probability tensor and probability dictionary
            nodes.append(model_util.make_zipmap_node(context, probability_tensor_name,
                                                     predicted_probability_name, class_labels))

        return nodes


# Register the class for processing
register_converter("glmClassifier", GLMClassifierConverter)

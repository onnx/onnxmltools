#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration


class ActivationConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'activation')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        activation_type = cm_node.activation.WhichOneof('NonlinearityType')
        params = cm_node.activation
        if activation_type == 'leakyReLU':
            nb = NodeBuilder(context, 'LeakyRelu')
            nb.add_attribute('alpha', params.leakyReLU.alpha)
        elif activation_type == 'ReLU':
            nb = NodeBuilder(context, 'Relu')
        elif activation_type == 'PReLU':
            nb = NodeBuilder(context, 'PRelu')
            nb.add_attribute('slope', params.PReLU.alpha)
        elif activation_type == 'ELU':
            nb = NodeBuilder(context, 'Elu')
            nb.add_attribute('alpha', params.ELU.alpha)
        elif activation_type == 'thresholdedReLU':
            nb = NodeBuilder(context, 'ThresholdedRelu')
            nb.add_attribute('alpha', params.thresholdedReLU.alpha)
        elif activation_type == 'tanh':
            nb = NodeBuilder(context, 'Tanh')
        elif activation_type == 'scaledTanh' :
            nb = NodeBuilder(context, 'ScaledTanh')
            nb.add_attribute('alpha', params.scaledTanh.alpha)
            nb.add_attribute('beta', params.scaledTanh.beta)
        elif activation_type == 'linear' :
            nb = NodeBuilder(context, 'Affine')
            nb.add_attribute('alpha', params.linear.alpha)
            nb.add_attribute('beta', params.linear.beta)
        elif activation_type == 'sigmoid':
            nb = NodeBuilder(context, 'Sigmoid')
        elif activation_type == 'sigmoidHard':
            nb = NodeBuilder(context, 'HardSigmoid')
            nb.add_attribute('alpha', params.sigmoidHard.alpha)
            nb.add_attribute('beta', params.sigmoidHard.beta)
        elif activation_type == 'softsign':
            nb = NodeBuilder(context, 'Softsign')
        elif activation_type == 'softplus':
            nb = NodeBuilder(context, 'Softplus')
        elif activation_type == 'parametricSoftplus':
            nb = NodeBuilder(context, 'Softplus')
            nb.add_attribute('alpha', params.parametricSoftplus.alpha)
            nb.add_attribute('beta', params.parametricSoftplus.beta)
        else:
            raise TypeError('Unsupported activation layer {0}'.format(activation_type))
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


# Register the class for processing
registration.register_nn_converter("activation", ActivationConverter)

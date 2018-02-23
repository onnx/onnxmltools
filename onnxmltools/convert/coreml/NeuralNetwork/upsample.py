#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import registration
from ...common import utils
from .reshape import extend_inputs_from_2d_to_4d
from coremltools.proto.NeuralNetwork_pb2 import UpsampleLayerParams as Params


class UpsampleLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'upsample')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        params = cm_node.upsample
        nb = NodeBuilder(context, 'Upsample')
        if params.mode == Params.NN:
            nb.add_attribute('mode', 'NEAREST')
        elif params.mode == Params.BILINEAR:
            nb.add_attribute('mode', 'BILINEAR')
        else:
            raise ValueError('Unsupported interpolation mode')
        scale_h = params.scalingFactor[0]
        scale_w = params.scalingFactor[1]
        nb.add_attribute('height_scale', float(scale_h) if scale_h != 0 else 1.)
        nb.add_attribute('width_scale', float(scale_w) if scale_w != 0 else 1.)
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


registration.register_nn_converter('upsample', UpsampleLayerConverter)

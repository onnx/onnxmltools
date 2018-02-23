#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d
from coremltools.proto.NeuralNetwork_pb2 import ReorganizeDataLayerParams as Params


class ReorganizeDataLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'reorganizeData')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)
        params = cm_node.reorganizeData
        if params.mode == Params.DEPTH_TO_SPACE:
            op_type = 'BatchToSpace'
        elif params.mode == Params.SPACE_TO_DEPTH:
            op_type = 'SpaceToBatch'
        else:
            raise ValueError('Unsupported reorganization mode {0}'.format(params.mode))

        nb = NodeBuilder(context, op_type)
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)
        nb.add_attribute('blocksize', params.blockSize)

        return nb.make_node()


registration.register_nn_converter('reorganizeData', ReorganizeDataLayerConverter)

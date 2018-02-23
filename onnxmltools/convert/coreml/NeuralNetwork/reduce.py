#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d
from coremltools.proto.NeuralNetwork_pb2 import ReduceLayerParams as Params


class ReduceLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'reduce')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        reduce_mode_map = {Params.SUM: 'ReduceSum', Params.AVG: 'ReduceMean', Params.PROD: 'ReduceProd',
                           Params.LOGSUM: 'ReduceLogSum', Params.SUMSQUARE: 'ReduceSumSquare',
                           Params.L1: 'ReduceL1', Params.L2: 'ReduceL2', Params.MAX: 'ReduceMax',
                           Params.MIN: 'ReduceMin', Params.ARGMAX: 'ArgMax'}

        params = cm_node.reduce
        reduce_mode = reduce_mode_map[params.mode]
        # CoreML's reduce operator is used to process tensors with shape [C, H, W]. Notice that [C, H, W] in CoreML
        # corresponds to [N, C, H, W] in ONNX because ONNX explicitly get the batch axis. If a CoreML reduce is working
        # on CoreML's C-axis, the corresponding ONNX axis's index would be 1 (for the 2nd axis in [N, C, H, W]-system).
        reduce_axis_map = {Params.CHW: [1, 2, 3], Params.HW: [2, 3], Params.C: [1], Params.H: [2], Params.W: [3]}
        reduce_axis = reduce_axis_map[params.axis]
        nb = NodeBuilder(context, reduce_mode)
        nb.add_attribute('axes', reduce_axis)
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


registration.register_nn_converter('reduce', ReduceLayerConverter)

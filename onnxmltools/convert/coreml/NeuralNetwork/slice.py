#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d
from coremltools.proto.NeuralNetwork_pb2 import SliceLayerParams as Params


class SliceLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'slice')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        nb = NodeBuilder(context, 'Slice')
        params = cm_node.slice

        axis_map = {Params.CHANNEL_AXIS: 0, Params.HEIGHT_AXIS: 1, Params.WIDTH_AXIS: 2}
        starts = [0, 0, 0]
        ends = [-1, -1, -1]

        starts[axis_map[params.axis]] = params.startIndex
        ends[axis_map[params.axis]] = params.endIndex

        # The input shape should be [N, C, H, W] in ONNX. Because CoreML only slices one of C-, H-, or W-axes, the
        # "axes" attribute in ONNX is [1, 2, 3]. Note that for the axes not really sliced, their starting and ending
        # indexes are 0 and -1, respectively.
        nb.add_attribute('axes', [1, 2, 3])
        nb.add_attribute('starts', starts)
        nb.add_attribute('ends', ends)
        if params.stride > 1:
            raise ValueError('Only unit stride is supported')
        nb.add_attribute('stride', params.stride)

        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


registration.register_nn_converter('slice', SliceLayerConverter)

#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common import NodeBuilder
from ...common import utils
from ...common import model_util
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d


def deduce_broadcast_axis_and_shape(shape):
    if len(shape) == 1:
        if shape[0] == 1:
            # shape is [1]
            return None, [1]
        else:
            # shape is [C]
            return 1, shape
    elif len(shape) == 3:
        if shape[0] == 1:
            # shape is [1, H, W]
            return 2, [shape[1], shape[2]]
        else:
            # shape is [C, H, W]
            return 1, shape


class BiasLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'bias')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        params = cm_node.bias

        nb = NodeBuilder(context, 'Add')
        # Feed the input (which we are going to add a bias onto) into Add operator. Its shape is [C, H, W] in CoreML but
        # [N, C, H, W] in ONNX.
        nb.extend_inputs(inputs)
        # Adjust CoreML's bias shape and find a proper axis for broadcasting
        axis, shape = deduce_broadcast_axis_and_shape(params.shape)
        # Create bias vector as an ONNX tensor
        tensor = model_util.make_tensor('B', onnx_proto.TensorProto.FLOAT, shape, params.bias.floatValue)
        # Create initializer and connect the initializer with Add's second input
        nb.add_initializer(tensor)
        # Tell Add the location to put its computation result
        nb.extend_outputs(outputs)
        # Add axis if the bias is not a scalar. If it's a scalar, broadcasting is naive.
        if axis is not None:
            nb.add_attribute('axis', axis)
        # No matter what shape it is, we need "broadcast" on because input shape is 4-D while bias is at most 3-D.
        nb.add_attribute('broadcast', 1)

        return nb.make_node()


registration.register_nn_converter('bias', BiasLayerConverter)

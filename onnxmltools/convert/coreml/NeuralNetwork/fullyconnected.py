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


class InnerProductLayerConverter:
    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'innerProduct')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        nb = NodeBuilder(context, 'FC')
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        params = cm_node.innerProduct
        linear_tensor = model_util.make_tensor('W', onnx_proto.TensorProto.FLOAT,
                                               [params.outputChannels, params.inputChannels], params.weights.floatValue)
        nb.add_initializer(linear_tensor)
        if params.hasBias:
            bias_tensor = model_util.make_tensor('b', onnx_proto.TensorProto.FLOAT, [params.outputChannels],
                                                 params.bias.floatValue)
        else:
            bias_tensor = model_util.make_tensor('b', onnx_proto.TensorProto.FLOAT, [params.outputChannels],
                                                 [0.] * params.outputChannels)
        nb.add_initializer(bias_tensor)
        nb.add_attribute('axis', 1)
        nb.add_attribute('axis_w', 1)

        return nb.make_node()


registration.register_nn_converter('innerProduct', InnerProductLayerConverter)

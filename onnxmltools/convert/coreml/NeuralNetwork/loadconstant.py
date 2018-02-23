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


class LoadConstantLayerConverter:
    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'loadConstant')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        nb = NodeBuilder(context, 'Constant')
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        constant = model_util.make_tensor('constant', onnx_proto.TensorProto.FLOAT,
                                          cm_node.loadConstant.shape, cm_node.loadConstant.data.floatValue)
        nb.add_attribute('value', constant)

        return nb.make_node()


registration.register_nn_converter('loadConstant', LoadConstantLayerConverter)

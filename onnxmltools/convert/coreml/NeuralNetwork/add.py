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


class AddLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'add')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        nb = NodeBuilder(context, 'Add')
        # Due to the limited support of broadcasting in ONNX, inputs should have identical shapes.
        nb.add_attribute('broadcast', 0)
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)
        if len(inputs) == 1:
            tensor = model_util.make_tensor('bias', onnx_proto.TensorProto.FLOAT, [len(cm_node.add.alpha)], cm_node.add.alpha)
            nb.add_initializer(tensor)

        return nb.make_node()


registration.register_nn_converter('add', AddLayerConverter)

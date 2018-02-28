#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration


class SplitLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'split')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        # ONNX Split may evenly divide the input along the specified axis if "split" attribute is not specified.
        # Also, CoreML always evenly split the input. Consequently, we only need to specify the axis and make sure the
        # number of outputs in ONNX matches that in CoreML.
        nb = NodeBuilder(context, 'Split', op_version=2)
        # CoreML's SplitLayer only works on the C-axis, so the axis index to cut is always 1.
        nb.add_attribute('axis', 1)
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


registration.register_nn_converter('split', SplitLayerConverter)

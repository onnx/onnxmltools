#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration


class L2NormalizeLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'l2normalize')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        nb = NodeBuilder(context, 'LpNormalization')
        # The first dimension is batch size, so the normalization is done along the 2nd axis (indexed by 1).
        nb.add_attribute('axis', 1)
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


registration.register_nn_converter('l2normalize', L2NormalizeLayerConverter)

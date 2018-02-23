#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d
from coremltools.proto.NeuralNetwork_pb2 import FlattenLayerParams as Params

class FlattenLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'flatten')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)
        nb = None
        if cm_node.flatten.mode == Params.CHANNEL_LAST:
            nb = NodeBuilder(context, 'Transpose')
            nb.extend_inputs(inputs)
            nb.add_output(nb.name)
            nb.add_attribute('perm', [0, 2, 3, 1])

        nb1 = NodeBuilder(context, 'Flatten')
        nb1.extend_inputs(inputs if nb is None else nb.output_names)
        nb1.extend_outputs(outputs)
        nb1.add_attribute('axis', 1)

        return [n.make_node() for n in [nb, nb1] if n is not None]


registration.register_nn_converter('flatten', FlattenLayerConverter)

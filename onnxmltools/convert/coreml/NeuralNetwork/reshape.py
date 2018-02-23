#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .simpleRecurrent import extract_dims
from coremltools.proto.NeuralNetwork_pb2 import ReshapeLayerParams as Params


def extend_inputs_from_2d_to_4d(context, inputs):
    # By default, input tensor [C]-tensor may be mapped to ['None', C]-tensor in ONNX. However, some
    # operators in CoreML require a 4-D input, so we need to change ['None', C] to ['None', C, 1, 1].
    for top_level_input in context.top_level_inputs:
        if context.get_onnx_name(top_level_input.name) != inputs[0]:
            continue
        dims = extract_dims(top_level_input)
        # The shape can be either ['None', C] or ['None', C, H, W] but we only make changes in the first
        # case. It also means that 2-D tensor would be adjusted at most once.
        if len(dims) != 2:
            continue
        dims += [1, 1] # dims becomes ['None', C, 1, 1]
        top_level_input.type.tensor_type.shape.dim.add()
        top_level_input.type.tensor_type.shape.dim.add()
        for i, d in enumerate(dims):
            if isinstance(d, str):
                top_level_input.type.tensor_type.shape.dim[i].dim_param = d
            else:
                top_level_input.type.tensor_type.shape.dim[i].dim_value = d


class ReshapeLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'reshape')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        nb = None
        params = cm_node.reshape
        if params.mode == Params.CHANNEL_LAST:
            nb = NodeBuilder(context, 'Transpose')
            nb.extend_inputs(inputs)
            nb.add_output(nb.name)
            nb.add_attribute('perm', [0, 2, 3, 1])

        nb1 = NodeBuilder(context, 'Reshape')
        nb1.add_attribute('shape', params.targetShape)
        nb1.extend_inputs(inputs if nb is None else nb.output_names)
        nb1.extend_outputs(outputs)

        return [n.make_node() for n in [nb, nb1] if n is not None]


registration.register_nn_converter('reshape', ReshapeLayerConverter)

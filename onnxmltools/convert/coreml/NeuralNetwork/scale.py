#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from ...common import model_util
from .bias import deduce_broadcast_axis_and_shape
from .reshape import extend_inputs_from_2d_to_4d


class ScaleLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'scale')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    # In CoreML's ScaleLayer, the input is first scaled by their "scale" attribute and then a bias can be added.
    # Symbols:
    #  a: scale attribute in CoreML's ScaleLayer
    #  b: bias attribute in CoreML's ScaleLayer
    #  x: input
    #  y: output
    # The math formulation of ScaleLayer should be
    #  y = a * x + b
    # Therefore, our strategy of composing ScaleLayer is to have one multiplication followed by an addition.
    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        params = cm_node.scale

        # Create a element-wise multiplication and use it to scale the input
        nb1 = NodeBuilder(context, 'Mul')
        # Set the original input as the first input of multiplication
        nb1.extend_inputs(inputs)
        # Create ONNX tensor to store the "scale" in CoreML
        axis_s, shape_s = deduce_broadcast_axis_and_shape(params.shapeScale)
        tensor_s = model_util.make_tensor('scale', onnx_proto.TensorProto.FLOAT, shape_s, params.scale.floatValue)
        # Set the factors used to scale the original input as the second input of multiplication
        nb1.add_initializer(tensor_s)
        if axis_s is not None:
            nb1.add_attribute('axis', axis_s)
        # No matter what shape it is, we need "broadcast" on because input shape is [N, C, H, W] and the "scale" in
        # CoreML is at most 3-D.
        nb1.add_attribute('broadcast', 1)

        if not params.hasBias:
            # If there is no bias to add, we directly output the result obtained from the multiplication.
            nb1.extend_outputs(outputs)

            return nb1.make_node()
        else:
            # If bias exists, we add the bias into the output of the multiplication and then use the output of addition
            # as the final output of this conversion.
            nb1.add_output(nb1.name)
            nb2 = NodeBuilder(context, 'Add')
            nb2.extend_inputs(nb1.output_names)
            axis_b, shape_b = deduce_broadcast_axis_and_shape(params.shapeScale)
            tensor_b = model_util.make_tensor('B', onnx_proto.TensorProto.FLOAT, shape_b, params.bias.floatValue)
            nb2.add_initializer(tensor_b)
            if axis_b is not None:
                nb2.add_attribute('axis', axis_b)
            # No matter what shape it is, we need "broadcast" on because input shape is [N, C, H, W] and the "bias" in
            # CoreML is at most 3-D.
            nb2.add_attribute('broadcast', 1)
            nb2.extend_outputs(outputs)

            return [nb1.make_node(), nb2.make_node()]


registration.register_nn_converter('scale', ScaleLayerConverter)

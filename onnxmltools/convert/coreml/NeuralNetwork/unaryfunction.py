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
from coremltools.proto.NeuralNetwork_pb2 import UnaryFunctionLayerParams as Params


class UnaryFunctionLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'unary')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        params = cm_node.unary
        nb1 = NodeBuilder(context, 'Affine')
        nb1.extend_inputs(inputs)
        nb1.add_output(nb1.name)

        nb1.add_attribute('alpha', params.scale if params.scale != 0 else 1.)
        nb1.add_attribute('beta', params.shift if params.shift != 0 else 0.)

        simple_unary_map = {Params.SQRT: 'Sqrt', Params.INVERSE: 'Reciprocal',
                            Params.EXP: 'Exp', Params.LOG: 'Log', Params.ABS: 'Abs'}
        if params.type == Params.RSQRT:
            nb2 = NodeBuilder(context, 'Sqrt')
            nb2.extend_inputs(nb1.output_names)
            nb2.add_output(nb2.name)
            nb3 = NodeBuilder(context, 'Reciprocal')
            nb3.extend_inputs(nb2.output_names)
            nb3.extend_outputs(outputs)
            return [nb1.make_node(), nb2.make_node(), nb3.make_node()]
        elif params.type == Params.POWER:
            nb2 = NodeBuilder(context, 'Pow')
            nb2.extend_inputs(nb1.output_names)
            tensor = model_util.make_tensor('Y', onnx_proto.TensorProto.FLOAT, [1], [params.alpha])
            nb2.add_initializer(tensor)
            nb2.extend_outputs(outputs)
            return [nb1.make_node(), nb2.make_node()]
        elif params.type == Params.THRESHOLD:
            nb2 = NodeBuilder(context, 'Clip')
            nb2.add_attribute('max', params.alpha)
            nb2.extend_inputs(nb1.output_names)
            nb2.extend_outputs(outputs)
            return [nb1.make_node(), nb2.make_node()]
        elif params.type in simple_unary_map:
            nb2 = NodeBuilder(context, simple_unary_map[params.type])
            nb2.extend_inputs(nb1.output_names)
            nb2.extend_outputs(outputs)
            return [nb1.make_node(), nb2.make_node()]
        else:
            raise ValueError('Unsupported unary function :{}'.format(params.type))


registration.register_nn_converter('unary', UnaryFunctionLayerConverter)

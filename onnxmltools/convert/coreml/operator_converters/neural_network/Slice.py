# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_converter
from .....proto import onnx_proto


def convert_slice(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import SliceLayerParams as Params

    op_type = 'Slice'
    op_name = scope.get_unique_operator_name(op_type)
    attrs = {'name': op_name}
    params = operator.raw_operator.slice

    # Set up slice range of C-, H-, and W-axes. Notice that only one of them will be actually sliced.
    axis_map = {Params.CHANNEL_AXIS: 0, Params.HEIGHT_AXIS: 1, Params.WIDTH_AXIS: 2}
    starts = [0, 0, 0]
    ends = [-1, -1, -1]
    starts[axis_map[params.axis]] = params.startIndex
    ends[axis_map[params.axis]] = params.endIndex

    if params.stride != 1:
        raise ValueError('Stride must be 1 but got %s' % params.stride)

    if container.target_opset < 10:
        # The input shape should be [N, C, H, W] in ONNX. Because CoreML only slices one of C-, H-, or W-axes, the
        # "axes" attribute in ONNX is [1, 2, 3]. Note that for the axes not really sliced, their starting and ending
        # indexes are 0 and -1, respectively.
        attrs['axes'] = [1, 2, 3]
        attrs['starts'] = starts
        attrs['ends'] = ends
        op_version = 1
        container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_version=op_version, **attrs)
    else:
        starts_name = scope.get_unique_variable_name(operator.full_name + '_starts')
        container.add_initializer(starts_name, onnx_proto.TensorProto.INT64,
                                  [3], starts)
        ends_name = scope.get_unique_variable_name(operator.full_name + '_ends')
        container.add_initializer(ends_name, onnx_proto.TensorProto.INT64,
                                  [3], ends)
        axes_name = scope.get_unique_variable_name(operator.full_name + '_axes')
        container.add_initializer(axes_name, onnx_proto.TensorProto.INT64,
                                  [3], [1, 2, 3])

        if container.target_opset < 11:
            op_version = 10
        else:
            op_version = 11
        container.add_node(op_type, [operator.input_full_names[0], starts_name, ends_name, axes_name], operator.output_full_names, op_version=op_version,
                           **attrs)


register_converter('slice', convert_slice)

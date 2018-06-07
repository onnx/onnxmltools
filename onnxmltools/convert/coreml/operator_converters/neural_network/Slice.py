# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_converter


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

    # The input shape should be [N, C, H, W] in ONNX. Because CoreML only slices one of C-, H-, or W-axes, the
    # "axes" attribute in ONNX is [1, 2, 3]. Note that for the axes not really sliced, their starting and ending
    # indexes are 0 and -1, respectively.
    attrs['axes'] = [1, 2, 3]
    attrs['starts'] = starts
    attrs['ends'] = ends
    if params.stride != 1:
        raise ValueError('Stride must be 1 but got %s' % params.stride)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_version=2, **attrs)


register_converter('slice', convert_slice)

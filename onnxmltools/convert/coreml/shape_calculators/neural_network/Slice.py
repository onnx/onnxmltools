# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common._data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_slice_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Slice has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.slice

    from coremltools.proto.NeuralNetwork_pb2 import SliceLayerParams as Params
    axis_map = {Params.CHANNEL_AXIS: 1, Params.HEIGHT_AXIS: 2, Params.WIDTH_AXIS: 3}

    if params.startIndex >= 0:
        output_shape[axis_map[Params.CHANNEL_AXIS]] = params.endIndex - params.startIndex
    else:
        output_shape[axis_map[Params.CHANNEL_AXIS]] += 1 + params.endIndex - params.startIndex

    operator.outputs[0].type = FloatTensorType(output_shape, doc_string=operator.outputs[0].type.doc_string)


register_shape_calculator('slice', calculate_slice_output_shapes)

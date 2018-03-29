# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_reduce_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Reduce has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    params = operator.raw_operator.reduce

    from coremltools.proto.NeuralNetwork_pb2 import ReduceLayerParams as Params
    # Adjust C-axis
    if params.axis in [Params.CHW, Params.C]:
        output_shape[1] = 1
    # Adjust H-axis
    if params.axis in [Params.CHW, Params.HW, Params.H]:
        output_shape[2] = 1
    # Adjust W-axis
    if params.axis in [Params.CHW, Params.HW, Params.W]:
        output_shape[3] = 1

    operator.outputs[0].type.shape = output_shape


register_shape_calculator('reduce', calculate_reduce_output_shapes)

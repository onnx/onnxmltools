# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_converter


def convert_reduce(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import ReduceLayerParams as Params

    reduce_mode_table = {
        Params.SUM: "ReduceSum",
        Params.AVG: "ReduceMean",
        Params.PROD: "ReduceProd",
        Params.LOGSUM: "ReduceLogSum",
        Params.SUMSQUARE: "ReduceSumSquare",
        Params.L1: "ReduceL1",
        Params.L2: "ReduceL2",
        Params.MAX: "ReduceMax",
        Params.MIN: "ReduceMin",
        Params.ARGMAX: "ArgMax",
    }
    params = operator.raw_operator.reduce
    reduce_mode = reduce_mode_table[params.mode]
    reduce_name = scope.get_unique_operator_name(reduce_mode)
    # CoreML's reduce operator is used to process tensors
    # with shape [C, H, W]. Notice that [C, H, W] in CoreML
    # corresponds to [N, C, H, W] in ONNX because ONNX explicitly
    # get the batch axis. If a CoreML reduce is working
    # on CoreML's C-axis, the corresponding ONNX axis's
    # index would be 1 (for the 2nd axis in [N, C, H, W]-system).
    reduce_axis_table = {
        Params.CHW: [1, 2, 3],
        Params.HW: [2, 3],
        Params.C: [1],
        Params.H: [2],
        Params.W: [3],
    }
    reduce_axis = reduce_axis_table[params.axis]
    attrs = {"name": reduce_name, "axes": reduce_axis}
    container.add_node(
        reduce_mode, operator.input_full_names, operator.output_full_names, **attrs
    )


register_converter("reduce", convert_reduce)

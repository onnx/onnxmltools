# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_converter


def convert_reorganize_data(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import ReorganizeDataLayerParams as Params

    params = operator.raw_operator.reorganizeData
    if params.mode == Params.DEPTH_TO_SPACE:
        op_type = "DepthToSpace"
    elif params.mode == Params.SPACE_TO_DEPTH:
        op_type = "SpaceToDepth"
    else:
        raise ValueError("Unsupported reorganization mode {0}".format(params.mode))

    op_name = scope.get_unique_operator_name(op_type)
    attrs = {"name": op_name, "blocksize": params.blockSize}
    container.add_node(
        op_type, operator.input_full_names, operator.output_full_names, **attrs
    )


register_converter("reorganizeData", convert_reorganize_data)

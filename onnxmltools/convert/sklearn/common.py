#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...proto import onnx_proto
from ..common import model_util
from ..common import NodeBuilder


def add_zipmap(input_name, key_type, class_labels, context):
    prob_output_name = context.get_unique_name(input_name)
    prob_value_type = onnx_proto.TensorProto.FLOAT
    prob_output = model_util.make_map_value_info(prob_output_name, key_type, prob_value_type)
    appended_node = model_util.make_zipmap_node(context, input_name, prob_output, class_labels)
    context.add_output(prob_output)
    return appended_node


def add_normalizer(input_name, output_type, norm, context):
    prob_output_name = context.get_unique_name(input_name)
    prob_output = model_util.make_tensor_value_info(prob_output_name, output_type)
    appended_node = model_util.make_normalizer_node(context, input_name, prob_output, norm)
    return appended_node, prob_output_name


def create_scaler(input, output_name, scale, offset, context):
    nb = NodeBuilder(context, "Scaler")
    nb.add_attribute('scale', [scale])
    nb.add_attribute('offset', [offset])

    nb.add_input(input)

    # Flatten out the input dims to create the tensor
    output_shape = [x.dim_value for x in input.type.tensor_type.shape.dim]
    output = model_util.make_tensor_value_info(context.get_unique_name(output_name),
                                               onnx_proto.TensorProto.FLOAT,
                                               output_shape)
    nb.add_output(output)
    return nb.make_node()

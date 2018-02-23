#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import sklearn
from ...proto import onnx_proto
from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from ..common import model_util


class LabelEncoderConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'classes_')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        nb = NodeBuilder(context, "LabelEncoder")
        nb.add_attribute('classes_strings', [str(c) for c in sk_node.classes_])
        nb.extend_inputs(inputs)
        try:
            if inputs[0].type.tensor_type.elem_type == onnx_proto.TensorProto.STRING:
                output_type = onnx_proto.TensorProto.INT64
                nb.add_attribute('default_int64', -1)
            elif inputs[0].type.tensor_type.elem_type == onnx_proto.TensorProto.INT64:
                output_type = onnx_proto.TensorProto.STRING
                nb.add_attribute('default_string', '__unknown__')
            else:
                raise ValueError()
        except AttributeError as e:
            raise ValueError('Invalid or missing input type for LabelEncoder.')
        try:
            output_dim = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
        except AttributeError as e:
            raise ValueError('Invalid or missing input dimension for LabelEncoder.')
        nb.add_output(model_util.make_tensor_value_info(nb.name, output_type, output_dim))

        return nb.make_node()


# Register the class for processing
register_converter(sklearn.preprocessing.LabelEncoder, LabelEncoderConverter)

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


class NormalizerConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'norm')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        nb = NodeBuilder(context, "Normalizer", op_domain='ai.onnx.ml')
        norm_enum = {'max': 'MAX', 'l1': 'L1', 'l2': 'L2'}
        if sk_node.norm in norm_enum.keys():
            nb.add_attribute('norm', norm_enum[sk_node.norm])
        else:
            raise RuntimeError("Invalid norm:" + sk_node.norm)

        nb.extend_inputs(inputs)
        try:
            output_dim = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
        except AttributeError as e:
            raise ValueError('Invalid or missing input dimension for Normalizer.')
        nb.add_output(model_util.make_tensor_value_info(nb.name, onnx_proto.TensorProto.FLOAT, output_dim))

        return nb.make_node()


# Register the class for processing
register_converter(sklearn.preprocessing.Normalizer, NormalizerConverter)

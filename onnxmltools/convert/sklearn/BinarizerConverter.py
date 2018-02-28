#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from ..common import model_util
import sklearn.preprocessing


class BinarizerConverter:

    @staticmethod
    def validate(node):
        try:
            utils._check_has_attr(node, 'threshold')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        nb = NodeBuilder(context, "Binarizer", op_domain='ai.onnx.ml')
        if isinstance(sk_node.threshold, list):
            raise ValueError(
                "Model which we try to convert contains multiple thresholds in Binarizer"
                "According to documentation only one threshold is allowed")
        nb.add_attribute('threshold', float(sk_node.threshold))

        nb.extend_inputs(inputs)
        try:
            output_type = inputs[0].type.tensor_type.elem_type
            output_dim = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
        except:
            raise ValueError('Invalid/missing input for Binarizer.')
        nb.add_output(model_util.make_tensor_value_info(nb.name, output_type, output_dim))

        return nb.make_node()


# Register the class for processing
register_converter(sklearn.preprocessing.Binarizer, BinarizerConverter)

#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
import sklearn.preprocessing
from ...proto import onnx_proto
from ..common import NodeBuilder
from ..common import register_converter
from ..common import utils
from ..common import model_util


class ImputerConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'statistics_')
            utils._check_has_attr(sk_node, 'missing_values')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):

        # Always use floats for the imputer -- to ensure this, any integer input
        # will be converted to a float using a scaler operation
        imputer_inputs = []
        nodes = []
        num_features = 0
        for inp in inputs:
            if inp.type.tensor_type.elem_type in model_util.onnx_integer_types:
                # Add the scaler node for int-to-float conversion
                scaler = model_util.create_scaler(inp, inp.name, 1.0, 0.0, context)
                nodes.append(scaler)
                imputer_inputs.append(scaler.outputs[0])
            else:
                imputer_inputs.append(inp)
            num_features += model_util.get_feature_count(imputer_inputs[-1])

        nb = NodeBuilder(context, 'Imputer', op_domain='ai.onnx.ml')
        nb.add_attribute('imputed_value_floats', sk_node.statistics_)

        replaced_value = 0.0
        if isinstance(sk_node.missing_values, str):
            if sk_node.missing_values == 'NaN':
                replaced_value = np.NaN
        elif isinstance(sk_node.missing_values, float):
            replaced_value = float(sk_node.missing_values)
        else:
            raise RuntimeError('Unsupported missing value')
        nb.add_attribute('replaced_value_float', replaced_value)

        nb.extend_inputs(imputer_inputs)
        nb.add_output(model_util.make_tensor_value_info(nb.name, onnx_proto.TensorProto.FLOAT, [1, num_features]))
        nodes.append(nb.make_node())

        return nodes


# Register the class for processing
register_converter(sklearn.preprocessing.Imputer, ImputerConverter)

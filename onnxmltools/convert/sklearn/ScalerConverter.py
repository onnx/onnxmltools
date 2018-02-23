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


class ScalerConverter():

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'scale_')
            utils._check_has_attr(sk_node, 'mean_')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))
        # Scikit converts zeroes in scale explicitly so all components should have non-zero values.
        if (sk_node.scale_ == 0).any():
            raise RuntimeError("The scale cannot contain any zero")

    @staticmethod
    def convert(context, sk_node, inputs):
        scale = 1.0 / sk_node.scale_
        offset = sk_node.mean_
        nb = NodeBuilder(context, "Scaler")
        nb.add_attribute('scale', scale)
        nb.add_attribute('offset', offset)

        nb.extend_inputs(inputs)
        try:
            output_dim = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
        except AttributeError as e:
            raise ValueError('Invalid/missing input dimension for Scaler.')
        nb.add_output(model_util.make_tensor_value_info(nb.name, onnx_proto.TensorProto.FLOAT, output_dim))

        return nb.make_node()


register_converter(sklearn.preprocessing.StandardScaler, ScalerConverter)

#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import sklearn.feature_extraction
from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from ..common import model_util
from ...proto import onnx_proto


class DictVectorizerConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'feature_names_')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        string_vocabulary = []
        int64_vocabulary = []
        key_type = value_type = None
        nb = NodeBuilder(context, 'DictVectorizer')
        for feature_name in sk_node.feature_names_:
            if utils.is_string_type(feature_name):
                string_vocabulary.append(feature_name)
                key_type = onnx_proto.TensorProto.STRING
                value_type = onnx_proto.TensorProto.FLOAT
            elif utils.is_numeric_type(feature_name):
                int64_vocabulary.append(feature_name)
                key_type = onnx_proto.TensorProto.INT64
                value_type = onnx_proto.TensorProto.FLOAT
            else:
                raise ValueError("Invalid or unsupported DictVectorizer type.")

        if len(string_vocabulary) > 0:
            nb.add_attribute('string_vocabulary', string_vocabulary)

        if len(int64_vocabulary) > 0:
            nb.add_attribute('int64_vocabulary', int64_vocabulary)

        nb.extend_inputs(inputs)
        nb.add_output(model_util.make_tensor_value_info(nb.name, value_type, [len(sk_node.feature_names_)]))

        return nb.make_node()


# Register the class for processing
register_converter(sklearn.feature_extraction.DictVectorizer, DictVectorizerConverter)

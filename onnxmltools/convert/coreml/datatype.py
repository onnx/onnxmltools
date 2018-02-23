#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...proto import onnx_proto
from ..common import model_util

COREML_TYPE_TO_ONNX_TYPE = {
    'int64Type': onnx_proto.TensorProto.INT64,
    'doubleType': onnx_proto.TensorProto.FLOAT,
    'stringType': onnx_proto.TensorProto.STRING,
    'int64KeyType': onnx_proto.TensorProto.INT64,
    'stringKeyType': onnx_proto.TensorProto.STRING,
    65568: onnx_proto.TensorProto.FLOAT,
    65600: onnx_proto.TensorProto.FLOAT,
    131104: onnx_proto.TensorProto.INT32
}

COREML_TREE_NODE_BEHAVIOR_TO_ONNX_TREE_NODE_MODE={
    0: 'BRANCH_LEQ',
    1: 'BRANCH_LT',
    2: 'BRANCH_GTE',
    3: 'BRANCH_GT',
    4: 'BRANCH_EQ',
    5: 'BRANCH_NEQ',
    6: 'LEAF'
}

COREML_TREE_POST_TRANSFORM_TO_ONNX_TREE_POST_TRANSFORM={
    0: 'NONE',
    1: 'SOFTMAX',
    2: 'LOGISTIC',
    3: 'SOFTMAX_ZERO'
}


def _convert(cm_type):
    if cm_type not in COREML_TYPE_TO_ONNX_TYPE:
        raise Exception("unsupported coreml type: " + cm_type)
    return COREML_TYPE_TO_ONNX_TYPE[cm_type]


def _handle_scalar_feature(cm_value):
    which_type = cm_value.type.WhichOneof('Type')
    onnx_type = _convert(which_type)
    onnx_shape = [1]
    return model_util.make_tensor_value_info(cm_value.name, onnx_type, onnx_shape)


def _handle_multi_array_feature(cm_value, batch_size=1):
    data_type = cm_value.type.multiArrayType.dataType
    onnx_type = _convert(data_type)
    onnx_shape = [batch_size]
    for shape_val in cm_value.type.multiArrayType.shape:
        onnx_shape.append(shape_val)
    return model_util.make_tensor_value_info(cm_value.name, onnx_type, onnx_shape)


def _handle_dictionary_feature(cm_value):
    key_type = cm_value.type.dictionaryType.WhichOneof('KeyType')
    onnx_key_type = _convert(key_type)
    onnx_value_type = onnx_proto.TensorProto.FLOAT
    map_type = model_util.make_map_value_info(cm_value.name, onnx_key_type, onnx_value_type)
    return map_type


def _handle_image_feature(cm_value, batch_size=1):
    # ONNX currently doesn't have image type, so we use tensor as images' representations.
    # One issue is that we are not able to add side information such as color space.
    onnx_type = onnx_proto.TensorProto.FLOAT
    if cm_value.type.imageType.colorSpace == 10:
        onnx_shape = [batch_size, 1]
        doc_string = 'Image(s) in gray scale. If there are N images, it is a 4-D tensor with shape [N, 1, H, W]'
    elif cm_value.type.imageType.colorSpace == 20:
        onnx_shape = [batch_size, 3]
        doc_string = 'Image(s) in RGB format. It is a [N, C, H, W]-tensor. The 1st/2nd/3rd slices along the' \
                     'C-axis are red, green, and blue channels, respectively.'
    elif cm_value.type.imageType.colorSpace == 30:
        onnx_shape = [batch_size, 3]
        doc_string = 'Image(s) in BGR format. It is a [N, C, H, W]-tensor. The 1st/2nd/3rd slices along the' \
                     'C-axis are blue, green, and red channels, respectively.'
    else:
        raise ValueError('Unsupported color space')
    onnx_shape.append(cm_value.type.imageType.height)
    onnx_shape.append(cm_value.type.imageType.width)
    return model_util.make_tensor_value_info(cm_value.name, onnx_type, onnx_shape, doc_string)


def make_value_info(value, batch_size=1):
    scalar_feature_types = {'int64Type', 'doubleType', 'stringType'}

    try:
        which_type = value.type.WhichOneof('Type')
    except Exception:
        raise Exception("Not a valid CoreML type")

    if which_type in scalar_feature_types:
        return _handle_scalar_feature(value)
    elif which_type == 'multiArrayType':
        return _handle_multi_array_feature(value, batch_size)
    elif which_type == 'dictionaryType':
        return _handle_dictionary_feature(value)
    elif which_type == 'imageType':
        return _handle_image_feature(value, batch_size)


def get_onnx_tree_mode(cm_tree_behavior):
    if cm_tree_behavior in COREML_TREE_NODE_BEHAVIOR_TO_ONNX_TREE_NODE_MODE:
        return COREML_TREE_NODE_BEHAVIOR_TO_ONNX_TREE_NODE_MODE[cm_tree_behavior]
    raise RuntimeError('CoreML tree node behavior not supported {0}'.format(cm_tree_behavior))


def get_onnx_tree_post_transform(cm_tree_post_transform):
    if cm_tree_post_transform in COREML_TREE_POST_TRANSFORM_TO_ONNX_TREE_POST_TRANSFORM:
        return COREML_TREE_POST_TRANSFORM_TO_ONNX_TREE_POST_TRANSFORM[cm_tree_post_transform]
    raise RuntimeError('CoreML tree post transform not supported {0}'.format(cm_tree_post_transform))

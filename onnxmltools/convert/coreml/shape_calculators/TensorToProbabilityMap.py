# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_shape_calculator
from ...common.data_types import (
    DictionaryType,
    FloatTensorType,
    SequenceType,
    StringTensorType,
    Int64TensorType,
)
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_tensor_to_probability_map_output_shapes(operator):
    """
    Allowed input/output patterns are
    ONNX < 1.2
        1. [1, C] ---> ---> A map
        2. [1, C_1, ..., C_n] ---> A map
    ONNX >= 1.2
        1. [N, C] ---> ---> A sequence of maps
        2. [N, C_1, ..., C_n] ---> A sequence of maps

    Note that N must be 1 currently if you're using
    ONNX<1.2 because old ZipMap doesn't produce a seqneuce of map If the
    input is not [N, C], it will be reshaped into
    [N, C_1 x C_2, x ... x C_n] before being fed into ONNX ZipMap.
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    model_type = operator.raw_operator.WhichOneof("Type")
    if model_type == "neuralNetworkClassifier":
        class_label_type = operator.raw_operator.neuralNetworkClassifier.WhichOneof(
            "ClassLabels"
        )
    else:
        raise TypeError("%s has no class label" % model_type)

    N = operator.inputs[0].type.shape[0]
    doc_string = operator.outputs[0].type.doc_string
    if class_label_type == "stringClassLabels":
        if operator.target_opset < 7:
            operator.outputs[0].type = DictionaryType(
                StringTensorType([1]), FloatTensorType([1]), doc_string
            )
        else:
            operator.outputs[0].type = SequenceType(
                DictionaryType(StringTensorType([]), FloatTensorType([])), N, doc_string
            )
    elif class_label_type == "int64ClassLabels":
        if operator.target_opset < 7:
            operator.outputs[0].type = DictionaryType(
                Int64TensorType([1]), FloatTensorType([1]), doc_string
            )
        else:
            operator.outputs[0].type = SequenceType(
                DictionaryType(Int64TensorType([]), FloatTensorType([])), N, doc_string
            )
    else:
        raise ValueError("Unsupported label type")


register_shape_calculator(
    "tensorToProbabilityMap", calculate_tensor_to_probability_map_output_shapes
)

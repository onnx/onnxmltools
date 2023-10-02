# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_shape_calculator
from ...common.data_types import (
    FloatTensorType,
    Int64TensorType,
    FloatType,
    Int64Type,
    DictionaryType,
    SequenceType,
    StringTensorType,
)
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_traditional_classifier_output_shapes(operator):
    """
    For classifiers, allowed input/output patterns are
        1. [N, C_1], ..., [N, C_n] ---> [N], Sequence of Map
        2. [N, C_1], ..., [N, C_n] ---> [N]

    For regressors, allowed input/output patterns are
        1. [N, C_1], ..., [N, C_n] ---> [N, 1]

    Core ML classifiers and regressors support multiple
    input feature tensors, so we need to concatenate them before
    feeding them into their ONNX counterparts.
    Note that the N must be 1 as long as ZipMap only produces dictionary.
    """
    check_input_and_output_numbers(
        operator, input_count_range=[1, None], output_count_range=[1, 2]
    )
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType, Int64TensorType, FloatType, Int64Type],
    )

    if any(len(variable.type.shape) != 2 for variable in operator.inputs):
        raise RuntimeError("Input(s) must be [N, C]-tensor(s)")

    model_type = operator.raw_operator.WhichOneof("Type")
    if model_type == "treeEnsembleClassifier":
        class_label_type = operator.raw_operator.treeEnsembleClassifier.WhichOneof(
            "ClassLabels"
        )
    elif model_type == "glmClassifier":
        class_label_type = operator.raw_operator.glmClassifier.WhichOneof("ClassLabels")
    elif model_type == "supportVectorClassifier":
        class_label_type = operator.raw_operator.supportVectorClassifier.WhichOneof(
            "ClassLabels"
        )
    else:
        raise ValueError("%s has no class label" % model_type)

    N = operator.inputs[0].type.shape[0]
    if operator.target_opset < 7:
        output_shape = [1, 1]
    else:
        output_shape = [N]

    if class_label_type == "stringClassLabels":
        operator.outputs[0].type = StringTensorType(
            output_shape, doc_string=operator.outputs[0].type.doc_string
        )
        if len(operator.outputs) == 2:
            if operator.target_opset < 7:
                operator.outputs[1].type = DictionaryType(
                    StringTensorType([1]),
                    FloatTensorType([1]),
                    doc_string=operator.outputs[1].type.doc_string,
                )
            else:
                operator.outputs[1].type = SequenceType(
                    DictionaryType(StringTensorType([]), FloatTensorType([])),
                    doc_string=operator.outputs[1].type.doc_string,
                )
    elif class_label_type == "int64ClassLabels":
        operator.outputs[0].type = Int64TensorType(
            output_shape, doc_string=operator.outputs[0].type.doc_string
        )
        if len(operator.outputs) == 2:
            if operator.target_opset < 7:
                operator.outputs[1].type = DictionaryType(
                    Int64TensorType([1]),
                    FloatTensorType([1]),
                    doc_string=operator.outputs[1].type.doc_string,
                )
            else:
                operator.outputs[1].type = SequenceType(
                    DictionaryType(Int64TensorType([]), FloatTensorType([])),
                    doc_string=operator.outputs[1].type.doc_string,
                )
    else:
        raise ValueError("Traditional classifier must include label information")


register_shape_calculator(
    "glmClassifier", calculate_traditional_classifier_output_shapes
)
register_shape_calculator(
    "supportVectorClassifier", calculate_traditional_classifier_output_shapes
)
register_shape_calculator(
    "treeEnsembleClassifier", calculate_traditional_classifier_output_shapes
)

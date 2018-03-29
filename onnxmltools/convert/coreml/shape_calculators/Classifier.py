# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common.data_types import FloatTensorType, Int64TensorType, FloatType, Int64Type, DictionaryType, StringType
from ...common._registration import register_shape_calculator


def calculate_traditional_classifier_output_shapes(operator):
    if len(operator.outputs) > 2 or len(operator.outputs) < 1:
        raise RuntimeError('Classifier cannot produce more than two or no output')

    if any(not isinstance(variable.type, (FloatTensorType, Int64TensorType, FloatType, Int64Type)) for variable in
           operator.inputs):
        raise RuntimeError('Input(s) must be tensor(s)')
    if any(len(variable.type.shape) != 2 or variable.type.shape[0] != 1 for variable in operator.inputs):
        raise RuntimeError('Input(s) must be [1,C]-tensor(s)')

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'treeEnsembleClassifier':
        class_label_type = operator.raw_operator.treeEnsembleClassifier.WhichOneof('ClassLabels')
    elif model_type == 'glmClassifier':
        class_label_type = operator.raw_operator.glmClassifier.WhichOneof('ClassLabels')
    elif model_type == 'supportVectorClassifier':
        class_label_type = operator.raw_operator.supportVectorClassifier.WhichOneof('ClassLabels')
    else:
        raise ValueError('%s has no class label' % model_type)

    if class_label_type == 'stringClassLabels':
        operator.outputs[0].type = StringType(doc_string=operator.outputs[0].type.doc_string)
        if len(operator.outputs) == 2:
            operator.outputs[1].type = DictionaryType(StringType(), FloatType(),
                                                      doc_string=operator.outputs[1].type.doc_string)
    elif class_label_type == 'int64ClassLabels':
        operator.outputs[0].type = Int64Type(doc_string=operator.outputs[0].type.doc_string)
        if len(operator.outputs) == 2:
            operator.outputs[1].type = DictionaryType(Int64Type(), FloatType(),
                                                      doc_string=operator.outputs[1].type.doc_string)
    else:
        raise ValueError('Traditional classifier must include label information')


register_shape_calculator('glmClassifier', calculate_traditional_classifier_output_shapes)
register_shape_calculator('supportVectorClassifier', calculate_traditional_classifier_output_shapes)
register_shape_calculator('treeEnsembleClassifier', calculate_traditional_classifier_output_shapes)

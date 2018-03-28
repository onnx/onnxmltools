import numpy as np
import six, numbers
from ...coreml._data_types import Int64TensorType, FloatTensorType, StringTensorType, DictionaryType
from ...coreml.registration import register_shape_calculator


def calculate_sklearn_linear_classifier_output_shapes(operator):
    if len(operator.inputs) != 1:
        raise RuntimeError('Only one input is allowed')
    if type(operator.inputs[0].type) not in [FloatTensorType, Int64TensorType]:
        raise RuntimeError('Only floats or integers can be input')
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]
    if N != 1:
        raise ValueError('Currently we only support one example per batch')

    class_labels = operator.raw_operator.classes_
    if all(isinstance(i, np.ndarray) for i in class_labels):
        class_labels = np.concatenate(class_labels)
    if all(isinstance(i, (six.string_types, six.text_type)) for i in class_labels):
        operator.outputs[0].type = StringTensorType(shape=[1, 1])
        if len(class_labels) > 2 or operator.type != 'SklearnLinearSVC':
            # For multi-class classifier, we produce a map for encoding the probabilities of all classes
            operator.outputs[1].type = DictionaryType(StringTensorType([1]), FloatTensorType([1]))
        else:
            # For binary classifier, we produce the probability of the positive class
            operator.outputs[1].type = FloatTensorType(shape=[1, 1])
    elif all(isinstance(i, numbers.Real) for i in class_labels):
        operator.outputs[0].type = Int64TensorType(shape=[1, 1])
        if len(class_labels) > 2 or operator.type != 'SklearnLinearSVC':
            # For multi-class classifier, we produce a map for encoding the probabilities of all classes
            operator.outputs[1].type = DictionaryType(Int64TensorType([1]), FloatTensorType([1]))
        else:
            # For binary classifier, we produce the probability of the positive class
            operator.outputs[1].type = FloatTensorType(shape=[1, 1])
    else:
        raise ValueError('Unsupported or mixed label types')


register_shape_calculator('SklearnLinearClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnLinearSVC', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnDecisionTreeClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnRandomForestClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnGradientBoostingClassifier', calculate_sklearn_linear_classifier_output_shapes)

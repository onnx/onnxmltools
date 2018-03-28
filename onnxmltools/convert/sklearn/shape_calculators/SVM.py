import six, numbers
from ...coreml._data_types import Int64TensorType, FloatTensorType, StringTensorType, DictionaryType
from ...coreml.registration import register_shape_calculator


def calculate_sklearn_svm_output_shapes(operator):
    if len(operator.inputs) < 1:
        raise RuntimeError('At least one input needs to present')

    op = operator.raw_operator

    N = operator.inputs[0].type.shape[0]

    if operator.type in ['SklearnSVC']:
        if N != 1 and N != 'None':
            # In this case, output probability map should be a sequence of dictionaries, which is not implemented yet.
            raise RuntimeError('Currently batch size must be one')
        if len(operator.outputs) != 2:
            raise RuntimeError('Support vector classifier has two outputs')
        if all(isinstance(i, (six.string_types, six.text_type)) for i in op.classes_):
            operator.outputs[0].type = StringTensorType([1, 1])
            operator.outputs[1].type = DictionaryType(StringTensorType([1]), FloatTensorType([1]))
        elif all(isinstance(i, numbers.Real) for i in op.classes_):
            operator.outputs[0].type = Int64TensorType([1, 1])
            operator.outputs[1].type = DictionaryType(Int64TensorType([1]), FloatTensorType([1]))
        else:
            raise RuntimeError('Class labels should be either all strings or all integers')

    if operator.type in ['SklearnSVR']:
        if len(operator.outputs) != 1:
            raise RuntimeError('Support vector regressor has only one output')

        operator.outputs[0].type = FloatTensorType([N, 1])


register_shape_calculator('SklearnSVC', calculate_sklearn_svm_output_shapes)
register_shape_calculator('SklearnSVR', calculate_sklearn_svm_output_shapes)

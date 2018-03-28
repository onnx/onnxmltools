import numbers
from ...coreml._data_types import FloatTensorType, Int64TensorType
from ...coreml.registration import register_shape_calculator


def calculate_sklearn_scaler_output_shapes(operator):
    # Inputs: multiple float- and integer-tensors
    # Output: one float tensor
    for variable in operator.inputs:
        if type(variable.type) not in [FloatTensorType, Int64TensorType]:
            raise RuntimeError('Scaler only accepts float- and int-tensors but got %s' % type(variable.type))
        if len(variable.type.shape) != 2:
            raise RuntimeError('Only 2-D tensor(s) can be input(s)')
        if len(set(variable.type.shape[0] for variable in operator.inputs)) > 1:
            raise RuntimeError('Batch size must be identical across inputs')
    if len(operator.outputs) != 1:
        raise RuntimeError('Only one output is allowed')
    if type(operator.outputs[0].type) != FloatTensorType:
        raise RuntimeError('Output must be a float tensor')

    N = variable.type.shape[0]
    C = 0
    for variable in operator.inputs:
        if isinstance(variable.type.shape[1], numbers.Integral):
            C += variable.type.shape[1]
        else:
            C = 'None'
            break

    operator.outputs[0].type.shape = [N, C]


register_shape_calculator('SklearnScaler', calculate_sklearn_scaler_output_shapes)
register_shape_calculator('SklearnNormalizer', calculate_sklearn_scaler_output_shapes)
register_shape_calculator('SklearnBinarizer', calculate_sklearn_scaler_output_shapes)

import copy
from ...common._data_types import Int64TensorType, StringTensorType
from ...common._registration import register_shape_calculator


def calculate_sklearn_lebel_encoder_output_shapes(operator):
    if len(operator.outputs) != 1:
        raise RuntimeError('Lebel encoder has only one output')

    if any(not isinstance(variable.type, (Int64TensorType, StringTensorType)) for variable in operator.inputs):
        raise RuntimeError('Unsupported input type(s) found')
    input_shape = copy.deepcopy(operator.inputs[0].type.shape)
    operator.outputs[0].type = Int64TensorType(copy.deepcopy(input_shape))


register_shape_calculator('SklearnLabelEncoder', calculate_sklearn_lebel_encoder_output_shapes)

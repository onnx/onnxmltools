from ...coreml._data_types import FloatTensorType
from ...coreml.registration import register_shape_calculator


def calculate_sklearn_linear_regressor_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('This is an one-to-one mapping')
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([N, 1])


register_shape_calculator('SklearnLinearRegressor', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnLinearSVR', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnDecisionTreeRegressor', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnRandomForestRegressor', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnGradientBoostingRegressor', calculate_sklearn_linear_regressor_output_shapes)

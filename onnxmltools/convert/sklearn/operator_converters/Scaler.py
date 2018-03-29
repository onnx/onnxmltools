from ...common._registration import register_converter
from .common import concatenate_variables


def convert_sklearn_scaler(scope, operator, container):
    # If there are multiple input variables, we need to combine them as a whole tensor. Integer(s) would be converted
    # to float(s).
    if len(operator.inputs) > 1:
        feature_name = concatenate_variables(scope, operator.inputs, container)
    else:
        feature_name = operator.inputs[0].full_name

    op_type = 'Scaler'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs['scale'] = 1.0 / operator.raw_operator.scale_
    attrs['offset'] = operator.raw_operator.mean_

    container.add_node(op_type, feature_name, operator.outputs[0].full_name, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnScaler', convert_sklearn_scaler)

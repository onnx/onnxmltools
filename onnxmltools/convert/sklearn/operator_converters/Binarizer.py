from ...coreml.registration import register_converter
from .common import concatenate_variables


def convert_sklearn_binarizer(scope, operator, container):
    if len(operator.inputs) > 1:
        # If there are multiple input tensors, we combine them using a FeatureVectorizer
        feature_name = concatenate_variables(scope, operator.inputs, container)
    else:
        # No concatenation is needed, we just use the first variable's name
        feature_name = operator.inputs[0].full_name

    op_type = 'Binarizer'
    attrs = {'name': scope.get_unique_operator_name(op_type), 'threshold': float(operator.raw_operator.threshold)}
    container.add_node(op_type, feature_name, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnBinarizer', convert_sklearn_binarizer)

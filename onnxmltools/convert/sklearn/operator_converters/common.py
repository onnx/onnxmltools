from ...coreml._data_types import Int64TensorType, Int64Type


def convert_integer_to_float(scope, variable, container):
    op_type = 'Scaler'
    scaled_name = scope.get_unique_variable_name(variable.full_name + '_scaled')
    scaler_attrs = {'name': scope.get_unique_operator_name(op_type), 'scale': [1.], 'offset': [0.]}
    container.add_node('Scaler', variable.full_name, scaled_name, op_domain='ai.onnx.ml', **scaler_attrs)
    return scaled_name


def concatenate_variables(scope, variables, container):
    '''
    This function allocate operators to from a float tensor by concatenating all input variables. Notice that if all
    integer inputs would be converted to floats before concatenation.
    '''
    op_type = 'FeatureVectorizer'
    attrs = {'name': scope.get_unique_operator_name(op_type)}

    input_names = []
    input_dims = []
    # Collect input variable names and do cast if needed
    for variable in variables:
        if type(variable.type) in [Int64TensorType, Int64Type]:
            # We use scaler to convert integers into floats because output is a single tensor and all tensor elements
            # should be in the same type.
            input_names.append(convert_integer_to_float(scope, variable, container))
        else:
            input_names.append(variable.full_name)
        # We assume input variables' shape are [1, C_1], ..., [1, C_n] if there are n inputs.
        input_dims.append(variable.type.shape[1])
    attrs['inputdimensions'] = input_dims

    # Create a variable name to capture feature vectorizer's output
    concatenated_name = scope.get_unique_variable_name('concatenated')
    container.add_node(op_type, input_names, concatenated_name, op_domain='ai.onnx.ml', **attrs)

    return concatenated_name

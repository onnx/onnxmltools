# SPDX-License-Identifier: Apache-2.0

from ...common.data_types import Int64TensorType, Int64Type, FloatTensorType, FloatType, StringType


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

    # Check if it's possible to concatenate those inputs.
    type_set = set(type(variable.type) for variable in variables)
    number_type_set = {FloatType, FloatTensorType, Int64Type, Int64TensorType}
    if StringType in type_set and any(number_type in type_set for number_type in number_type_set):
        raise RuntimeError('We are not able to concatenate numerical tensor(s) and string tensor(s)')

    input_names = []  # input variables' names we want to concatenate
    input_dims = []  # dimensions of the variables that is going to be concatenated

    # Collect input variable names and do cast if needed
    for variable in variables:
        if isinstance(variable.type, (Int64TensorType, Int64Type)):
            input_names.append(convert_integer_to_float(scope, variable, container))
        else:
            input_names.append(variable.full_name)
        # We assume input variables' shape are [1, C_1], ..., [1, C_n] if there are n inputs.
        input_dims.append(variable.type.shape[1])

    if len(input_names) == 1:
        # No need to concatenate tensors if there is only one input
        return input_names[0]
    else:
        # To combine all inputs, we need a FeatureVectorizer
        op_type = 'FeatureVectorizer'
        attrs = {'name': scope.get_unique_operator_name(op_type), 'inputdimensions': input_dims}
        # Create a variable name to capture feature vectorizer's output
        concatenated_name = scope.get_unique_variable_name('concatenated')
        # Set up our FeatureVectorizer
        container.add_node(op_type, input_names, concatenated_name, op_domain='ai.onnx.ml', **attrs)

        return concatenated_name

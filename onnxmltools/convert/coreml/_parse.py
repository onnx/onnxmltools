# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from . import operator_converters
from . import shape_calculators
from ._data_types import *
from ._topology import Topology
from .operator_converters import neural_network
from .shape_calculators import neural_network


def _parse_model(topology, scope, model, inputs=list(), outputs=list()):
    '''
    This is a delegate function of all top-level parsing functions. It does nothing but call a proper function
    to parse the given model.
    '''

    model_type = model.WhichOneof('Type')
    if model_type in ['pipeline', 'pipelineClassifier', 'pipelineRegressor']:
        _parse_pipeline_model(topology, scope, model, inputs, outputs)
    elif model_type in ['neuralNetworkClassifier', 'neuralNetworkRegressor', 'neuralNetwork']:
        _parse_neural_network_model(topology, scope, model, inputs, outputs)
    else:
        _parse_simple_model(topology, scope, model, inputs, outputs)


def _parse_simple_model(topology, parent_scope, model, inputs, outputs):
    '''
    Parse a model containing only one operator (aka simple model).
    Steps:
        1. Create local scope for allocating local variables and operators
        2. Create operator and then feed the model's inputs and outputs to the operator
        3. Connect local variables and their corresponding parent variables
    Note:
        1. Notice that a CoreML operator can contain no input and output, so we directly use model's inputs (outputs).
        2. Input and output names can be identical in CoreML, but they must be different for ONNX.
    '''

    # Create local scope for the considered model
    scope = topology.declare_scope('single', [parent_scope] + parent_scope.parent_scopes)

    # Create operator for the considered model
    this_operator = scope.declare_local_operator(model.WhichOneof('Type'), model)

    # Allocate inputs for the operator and then connect them with inputs from outside
    for var in model.description.input:
        # We assume that no duplicated raw name exists. Note that we set prepend=True because model inputs should
        # not hide any intermediate variables.
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature(var, topology.default_batch_size), prepend=True)
        this_operator.inputs.append(variable)

    # Connect local variables and variables passed into this scope. Our assumptions are described below.
    # 1. Assume a variable with 'A' as its CoreML name is passed in. There must be at least one local variable gets a
    #    raw name 'A'. That is, for each parent variable, at least one local duplicate is available.
    # 2. It's possible to find multiple local variables associated with the same raw name. For example, raw name 'A' can
    #    be associated with 'A' and 'A1' in ONNX. In this case, we connect the first one to parent input.
    for parent_variable in inputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][0]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(parent_variable)
        operator.outputs.append(child_variable)

    # Allocate outputs for the operator and then connect them with outputs from outside
    for var in model.description.output:
        # We assume that no duplicated output raw name exists.
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature(var, topology.default_batch_size))
        this_operator.outputs.append(variable)

    # Connect local variables and variables passed into this scope. Our assumptions are described below.
    # 1. Assume a variable with 'A' as its CoreML name is passed in. There must be at least one local variable gets a
    #    raw name 'A'. That is, for each parent variable, at least one local duplicate is available.
    # 2. It's possible to find multiple local variables associated with the same raw name. For example, raw name 'A' can
    #    be associated with 'A' and 'A1' in ONNX. In this case, we connect the last one to parent output.
    for parent_variable in outputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][-1]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(parent_variable)


def _parse_pipeline_model(topology, parent_scope, model, inputs, outputs):
    '''
    Parse a pipeline including multiple sub-models.
    Steps:
        1. Create local scope for allocating local variables and operators
        2. Sequentially parse the sub-models and create their inputs and outputs variables
        3. Connect model's (not sub-model's) inputs and outputs with proper variables created when parsing sub-models
        4. Link local variables and the corresponding parent variables (only model's inputs and outputs are considered)
    Note:
        1. A CoreML sub-model can use the same variable for its input and output.
        2. Two CoreML variables may have the same name but different types.
    '''

    # Create local scope
    scope = topology.declare_scope('pipeline', [parent_scope] + parent_scope.parent_scopes)

    # Use the same name to denote sub-models
    pipeline_type = model.WhichOneof('Type')
    if pipeline_type == 'pipelineClassifier':
        sub_models = model.pipelineClassifier.pipeline.models
    elif pipeline_type == 'pipelineRegressor':
        sub_models = model.pipelineRegressor.pipeline.models
    elif pipeline_type == 'pipeline':
        sub_models = model.pipeline.models
    else:
        raise ValueError('Unsupported CoreML pipeline type: {0}'.format(pipeline_type))

    # Sequentially parse the sub-models
    for sub_model in sub_models:
        # Declare the sub-model's input and output in this scope. Those input and output variables will be passed into
        # the sub-model's parsing function and connected with proper child variables.
        sub_inputs = []
        for var in sub_model.description.input:
            variable = scope.get_local_variable_or_declare_one(
                var.name, parse_coreml_feature(var, topology.default_batch_size))
            sub_inputs.append(variable)
        sub_outputs = []
        for var in sub_model.description.output:
            variable = scope.declare_local_variable(
                var.name, parse_coreml_feature(var, topology.default_batch_size))
            sub_outputs.append(variable)
        _parse_model(topology, scope, sub_model, sub_inputs, sub_outputs)

    # Declare the model's (not sub-model's) inputs and then link them with sub-model's inputs
    for var in model.description.input:
        # Find the first variable with the same raw name declared when parsing the sub-models
        child_variable = scope.variables[scope.variable_name_mapping[var.name][0]]
        # Create model's input variable. Note that we set prepend=True because model inputs should not hide any
        # intermediate variables.
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature(var, topology.default_batch_size), prepend=True)
        # Feed the input to the sub-model's input. It's possible to add type conversion here by using a casting operator
        # rather than identity, but we haven't see the need of doing so in practices.
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(variable)
        operator.outputs.append(child_variable)
    for parent_variable in inputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][0]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(parent_variable)
        operator.outputs.append(child_variable)

    # Declare the model's (not sub-model's) inputs and then link them with sub-model's inputs
    for var in model.description.output:
        # Find the latest variable with the same raw name declared when parsing the sub-models
        child_variable = scope.variables[scope.variable_name_mapping[var.name][-1]]
        # Create model's output variable
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature(var, topology.default_batch_size))
        # Connect the input and a sub-model's input. It's possible to add type conversion here by using a casting
        # operator rather than identity, but we haven't see the need of doing so in practices.
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(variable)
    for parent_variable in outputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][-1]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(parent_variable)


def _parse_neural_network_model(topology, parent_scope, model, inputs, outputs):
    '''
    Parse a neural network model.
    Steps:
        1. Create local scope for allocating local variables and operators
        2. Sequentially parse the preprocessors and layers
        3. Connect model's (neither layers' nor preprocessors') inputs and outputs with proper variables created when
           parsing sub-models.
        4. Link local variables and the corresponding parent variables (only model's inputs and outputs are considered)
    Note:
        1. A CoreML preprocessor/layer can use the same variable for its input and output.
        2. Two CoreML variables may have the same name but different types.
        3. Preprocessor sometime may not include any information about its input
    '''

    # Create local scope to which all subsequent variables and operators belongs
    scope = topology.declare_scope('NeuralNetwork', [parent_scope] + parent_scope.parent_scopes)

    network = None
    network_type = model.WhichOneof('Type')
    if network_type == 'neuralNetworkClassifier':
        network = model.neuralNetworkClassifier
    elif network_type == 'neuralNetworkRegressor':
        network = model.neuralNetworkRegressor
    elif network_type == 'neuralNetwork':
        network = model.neuralNetwork
    else:
        raise ValueError('Unknown network type {}'.format(network_type))

    for op in network.preprocessing:
        operator = scope.declare_local_operator(op.WhichOneof('preprocessor') + 'Preprocessor', op)

        # Infer the variable name to be processed if feature name is an empty string
        name = op.featureName if op.featureName != '' else model.description.input[0].name

        # Find out input variable
        original = scope.get_local_variable_or_declare_one(name)
        original.type = FloatTensorType()  # A newly-declared variable has no type, so we add it.
        operator.inputs.append(original)

        # Declare a variable for storing the processed result
        processed = scope.declare_local_variable(name)
        processed.type = FloatTensorType()  # A newly-declared variable has no type, so we add it
        operator.outputs.append(processed)

    for op in network.layers:
        operator = scope.declare_local_operator(op.WhichOneof('layer'), op)

        # Find out input variable and connect them with the operator
        for name in op.input:
            variable = scope.get_local_variable_or_declare_one(name)
            # Although most neural network operators only accepts floats, we still need to handle the only exception,
            # embedding layer. In the furture, we should create a Cast operator right inside embedding's converter.
            if operator.type == 'embedding':
                variable.type = Int64TensorType()
            else:
                variable.type = FloatTensorType()
            operator.inputs.append(variable)

        # Declare variables for catching the operator's outputs
        for name in op.output:
            variable = scope.declare_local_variable(name)
            variable.type = FloatTensorType()  # A newly-declared variable has no type, so we add it
            operator.outputs.append(variable)

    sink_variables = scope.find_sink_variables()

    # Declare the model's inputs and outputs. Then, connect them with proper variables computed by the main network
    for var in model.description.input:
        # Search for the first variable (declared when parsing network layers) associated with the considered raw name
        child_variable = scope.variables[scope.variable_name_mapping[var.name][0]]

        # Declare model input. To prevent intermediate variables form being hidden by model inputs, prepend is True.
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature(var, topology.default_batch_size), prepend=True)

        # A heuristic which forces the input of embedding to be integer tensor rather than float tensor.
        # Ideally this should be done by adding a cast operator, but ONNX doesn't have float-to-int casting.
        # If this variable is produced by another component in a CoreML pipeline, a bug may occur especially
        # when the source component's output type is float tensor.
        if isinstance(child_variable.type, Int64TensorType):
            variable.type = Int64TensorType(variable.type.shape)

        # Feed model input to the associated model input
        operator_type = find_type_conversion(source_type=variable.type, target_type=child_variable.type)
        operator = scope.declare_local_operator(operator_type)
        operator.inputs.append(variable)
        operator.outputs.append(child_variable)

    # Connect local input variables with proper variables from parent scope
    for parent_variable in inputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][0]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(parent_variable)
        operator.outputs.append(child_variable)

    for var in model.description.output:
        # CoreML's predicted label is not connected with any operator, so we handle it later as a special case.
        special_variable_names = [model.description.predictedFeatureName, model.description.predictedProbabilitiesName]
        if model.WhichOneof('Type') == 'neuralNetworkClassifier' and var.name in special_variable_names:
            continue
        # Search for the latest variable (declared when parsing network layers) associated with the considered raw name
        child_variable = scope.variables[scope.variable_name_mapping[var.name][-1]]

        # Create model output variable
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature(var, topology.default_batch_size))

        # Feed result calculated by the network to the output variable
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(variable)

    # If predicted label exists, connect probability tensor and label by a special operator
    if model.WhichOneof('Type') == 'neuralNetworkClassifier' and model.description.predictedFeatureName:
        # Find out the description of predicted label and declare a label variable
        label_variable = None
        for var in model.description.output:
            if var.name == model.description.predictedFeatureName:
                label_type = parse_coreml_feature(var, topology.default_batch_size)
                label_variable = scope.declare_local_variable(var.name, label_type)
                break
        operator = scope.declare_local_operator('tensorToLabel', model)

        probability_name = model.description.predictedProbabilitiesName
        if probability_name in scope.variable_name_mapping:
            # Find the latest probability variable
            operator.inputs.append(scope.variables[scope.variable_name_mapping[probability_name][-1]])
        else:
            # If predicted probability tensor is missing in CoreML model, it defaults to the first sink of the network
            operator.inputs.append(sink_variables[0])
        operator.outputs.append(label_variable)

    # Probability tensor is implicitly converted into a dictionary (i.e., map) in CoreML. We handle this case here.
    if model.WhichOneof('Type') == 'neuralNetworkClassifier' and model.description.predictedProbabilitiesName:
        operator = scope.declare_local_operator('tensorToProbabilityMap', model)

        probability_name = model.description.predictedProbabilitiesName
        if probability_name in scope.variable_name_mapping:
            # Find the latest probability variable
            operator.inputs.append(scope.variables[scope.variable_name_mapping[probability_name][-1]])
        else:
            # If predicted probability tensor is missing in CoreML model, it defaults to the first sink of the network
            operator.inputs.append(sink_variables[0])

        # Find out the description of predicted probabilities and declare a variable for probability map
        for var in model.description.output:
            if var.name == model.description.predictedProbabilitiesName:
                probability_type = parse_coreml_feature(var, topology.default_batch_size)
                probability_variable = scope.declare_local_variable(var.name, probability_type)
                operator.outputs.append(probability_variable)
                break

    # Connect local output variables with proper variables from parent scope
    for parent_variable in outputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][-1]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(parent_variable)


def parse_coreml(model, initial_types=dict()):
    '''
    This is the root function of the whole parsing procedure.
    :param model: CoreML model
    :param initial_types: a dictionary providing some types for some CoreML root variables. For example, a key-value
           pair, ('A', FloatTensorType([40, 12, 1, 1])), means that in your CoreML model, there is variable called 'A'
           and it's a float tensor with shape [40, 12, 1, 1].
    :return: a Topology object. It's a intermediate representation of the input CoreML model
    '''

    reserved_variable_names = set()
    for var in list(model.description.input) + list(model.description.output):
        reserved_variable_names.add(var.name)
    default_batch_size = 1 if model.WhichOneof('Type') not in \
                              ['neuralNetworkClassifier', 'neuralNetworkRegressor', 'neuralNetwork'] else 'None'
    topology = Topology(model, default_batch_size, initial_types, reserved_variable_names)
    scope = topology.declare_scope('__root__')
    _parse_model(topology, scope, model)
    topology.compile()
    for variable in topology.find_root_and_sink_variables():
        if variable.raw_name not in reserved_variable_names:
            continue
        topology.rename_variable(variable.onnx_name, variable.raw_name)
    return topology

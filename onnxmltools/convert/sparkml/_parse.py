# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .ops_names import get_sparkml_operator_name
from .ops_input_output import get_input_names, get_output_names

from ..common._container import SparkmlModelContainer
from ..common._topology import *

from pyspark.ml import PipelineModel


def _get_variable_for_input(scope, input_name, global_inputs, output_dict):
    '''
    Find the corresponding Variable for a given raw operator (model) name
    The variable is either supplied as graph/global inputs or has been generated as output by previous ops
    :param input_name:
    :param global_inputs:
    :param output_dict:
    :return:
    '''
    if input_name in output_dict:
        value = output_dict[input_name]
        ref_count = value[0]
        variable = value[1]
        output_dict[input_name] = [ref_count+1, variable]
        return variable

    matches = [x for x in global_inputs if x.raw_name == input_name]
    if matches:
        return matches[0]
    #
    # create a new Var
    #
    return scope.declare_local_variable(input_name)


def _parse_sparkml_simple_model(spark, scope, model, global_inputs, output_dict):
    '''
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A spark-ml Transformer/Evaluator (e.g., OneHotEncoder and LogisticRegression)
    :param global_inputs: A list of variables
    :param output_dict: An accumulated list of output_original_name->(ref_count, variable)
    :return: A list of output variables which will be passed to next stage
    '''
    this_operator = scope.declare_local_operator(get_sparkml_operator_name(type(model)), model)
    this_operator.raw_params = {'SparkSession': spark}
    raw_input_names = get_input_names(model)
    this_operator.inputs = [_get_variable_for_input(scope, x, global_inputs, output_dict) for x in raw_input_names]
    raw_output_names = get_output_names(model)
    for output_name in raw_output_names:
        variable = scope.declare_local_variable(output_name, FloatTensorType())
        this_operator.outputs.append(variable)
        output_dict[variable.raw_name] = [0, variable]


def _parse_sparkml_pipeline(spark, scope, model, global_inputs, output_dict):
    '''
    The basic ideas of spark-ml parsing:
        1. Sequentially go though all stages defined in the considered spark-ml pipeline
        2. The output variables of one stage will be fed into its next stage as the inputs.

    :param scope: Scope object defined in _topology.py
    :param model: spark-ml pipeline object
    :param global_inputs: A list of Variable objects
    :param output_dict: An accumulated list of output_original_name->(ref_count, variable)
    :return: A list of output variables produced by the input pipeline
    '''
    for stage in model.stages:
        _parse_sparkml(spark, scope, stage, global_inputs, output_dict)


def _parse_sparkml(spark, scope, model, global_inputs, output_dict):
    '''
    This is a delegate function. It doesn't nothing but invoke the correct parsing function according to the input
    model's type.
    :param scope: Scope object
    :param model: A spark-ml object (e.g., OneHotEncoder and LogisticRegression)
    :param inputs: A list of variables
    :return: The output variables produced by the input model
    '''
    if isinstance(model, PipelineModel):
        return _parse_sparkml_pipeline(spark, scope, model, global_inputs, output_dict)
    else:
        return _parse_sparkml_simple_model(spark, scope, model, global_inputs, output_dict)


def parse_sparkml(spark, model, initial_types=None, target_opset=None,
                  custom_conversion_functions=None, custom_shape_calculators=None):
    # Put spark-ml object into an abstract container so that our framework can work seamlessly on models created
    # with different machine learning tools.
    raw_model_container = SparkmlModelContainer(model)

    # Declare a computational graph. It will become a representation of the input spark-ml model after parsing.
    topology = Topology(raw_model_container, default_batch_size='None',
                        initial_types=initial_types,
                        target_opset=target_opset,
                        custom_conversion_functions=custom_conversion_functions,
                        custom_shape_calculators=custom_shape_calculators)

    # Declare an object to provide variables' and operators' naming mechanism. In contrast to CoreML, one global scope
    # is enough for parsing spark-ml models.
    scope = topology.declare_scope('__root__')

    # Declare input variables. They should be the inputs of the spark-ml model you want to convert into ONNX
    inputs = []
    for var_name, initial_type in initial_types:
        inputs.append(scope.declare_local_variable(var_name, initial_type))

    # The object raw_model_container is a part of the topology we're going to return. We use it to store the inputs of
    # the spark-ml's computational graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input spark-ml model as a Topology object.
    output_dict = {}
    _parse_sparkml(spark, scope, model, inputs, output_dict)
    outputs = []
    for k, v in output_dict.items():
        if v[0] == 0: # ref count is zero
            outputs.append(v[1])

    # THe object raw_model_container is a part of the topology we're going to return. We use it to store the outputs of
    # the spark-ml's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology

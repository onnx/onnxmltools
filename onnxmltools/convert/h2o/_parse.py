# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from onnxconverter_common.data_types import FloatTensorType
from ..common._container import H2OModelContainer
from ..common._topology import Topology

def _parse_h2o(scope, model, inputs):
    '''
    :param scope: Scope object
    :param model: A h2o model data object
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next stage
    '''
    this_operator = scope.declare_local_operator("H2OTreeMojo", model)
    this_operator.inputs = inputs

    if model["params"]["classifier"]:
        label_variable = scope.declare_local_variable('label', FloatTensorType())
        probability_map_variable = scope.declare_local_variable('probabilities', FloatTensorType())
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(probability_map_variable)
    else:
        variable = scope.declare_local_variable('variable', FloatTensorType())
        this_operator.outputs.append(variable)
    return this_operator.outputs


def parse_h2o(model, initial_types=None, target_opset=None,
              custom_conversion_functions=None, custom_shape_calculators=None):

    raw_model_container = H2OModelContainer(model)
    topology = Topology(raw_model_container, default_batch_size='None',
                        initial_types=initial_types, target_opset=target_opset,
                        custom_conversion_functions=custom_conversion_functions,
                        custom_shape_calculators=custom_shape_calculators)
    scope = topology.declare_scope('__root__')

    inputs = []
    for var_name, initial_type in initial_types:
        inputs.append(scope.declare_local_variable(var_name, initial_type))

    for variable in inputs:
        raw_model_container.add_input(variable)

    outputs = _parse_h2o(scope, model, inputs)

    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology

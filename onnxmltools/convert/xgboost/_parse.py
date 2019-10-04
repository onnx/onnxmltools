# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._container import XGBoostModelContainer
from ..common._topology import *

from xgboost import XGBRegressor, XGBClassifier

xgboost_classifier_list = [XGBClassifier]

# Associate types with our operator names.
xgboost_operator_name_map = {XGBClassifier: 'XGBClassifier',
                              XGBRegressor: 'XGBRegressor'}


def _get_xgboost_operator_name(model_type):
    '''
    Get operator name of the input argument

    :param model_type:  A xgboost object.
    :return: A string which stands for the type of the input model in our conversion framework
    '''
    if model_type not in xgboost_operator_name_map:
        raise ValueError("No proper operator name found for '%s'" % model_type)
    return xgboost_operator_name_map[model_type]


def _parse_xgboost_simple_model(scope, model, inputs):
    '''
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A xgboost object
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next stage
    '''
    this_operator = scope.declare_local_operator(_get_xgboost_operator_name(type(model)), model)
    this_operator.inputs = inputs

    if type(model) in xgboost_classifier_list:
        # For classifiers, we may have two outputs, one for label and the other one for probabilities of all classes.
        # Notice that their types here are not necessarily correct and they will be fixed in shape inference phase
        label_variable = scope.declare_local_variable('label', FloatTensorType())
        probability_map_variable = scope.declare_local_variable('probabilities', FloatTensorType())
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(probability_map_variable)
    else:
        # We assume that all scikit-learn operator can only produce a single float tensor.
        variable = scope.declare_local_variable('variable', FloatTensorType())
        this_operator.outputs.append(variable)
    return this_operator.outputs


def _parse_xgboost(scope, model, inputs):
    '''
    This is a delegate function. It doesn't nothing but invoke the correct parsing function according to the input
    model's type.
    :param scope: Scope object
    :param model: A xgboost object
    :param inputs: A list of variables
    :return: The output variables produced by the input model
    '''
    return _parse_xgboost_simple_model(scope, model, inputs)


def parse_xgboost(model, initial_types=None, target_opset=None,
                   custom_conversion_functions=None, custom_shape_calculators=None):

    raw_model_container = XGBoostModelContainer(model)
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

    outputs = _parse_xgboost(scope, model, inputs)

    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology

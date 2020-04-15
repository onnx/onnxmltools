# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from onnxconverter_common.data_types import FloatTensorType
from ..common._container import XGBoostModelContainer
from ..common._topology import Topology


xgboost_classifier_list = [XGBClassifier]

# Associate types with our operator names.
xgboost_operator_name_map = {XGBClassifier: 'XGBClassifier',
                              XGBRegressor: 'XGBRegressor'}


def _append_covers(node):
    res = []
    if 'cover' in node:
        res.append(node['cover'])
    if 'children' in node:
        for ch in node['children']:
            res.extend(_append_covers(ch))
    return res


def _get_attributes(booster):
    atts = booster.attributes()
    ntrees = booster.best_ntree_limit
    dp = booster.get_dump(dump_format='json', with_stats=True)        
    res = [json.loads(d) for d in dp]
    trees = len(res)
    kwargs = atts.copy()
    kwargs['feature_names'] = booster.feature_names
    kwargs['n_estimators'] = ntrees
    
    # covers
    covs = []
    for tr in res:
        covs.extend(_append_covers(tr))

    if all(map(lambda x: int(x) == x, set(covs))):
        # regression
        kwargs['num_class'] = 0
        if trees > ntrees > 0:
            kwargs['num_target'] = trees // ntrees
            kwargs["objective"] = "reg:squarederror"
        else:
            kwargs['num_target'] = 1
            kwargs["objective"] = "reg:squarederror"
    else:
        # classification
        kwargs['num_target'] = 0
        if trees > ntrees > 0:
            kwargs['num_class'] = trees // ntrees
            kwargs["objective"] = "multi:softprob"
        else:
            kwargs['num_class'] = 1
            kwargs["objective"] = "binary:logistic"

    if 'base_score' not in kwargs:
        kwargs['base_score'] = 0.5
    return kwargs


class WrappedBooster:

    def __init__(self, booster):
        self.booster_ = booster
        self.kwargs = _get_attributes(booster)

        if self.kwargs['num_class'] > 0:
            self.classes_ = self._generate_classes(self.kwargs)
            self.operator_name = 'XGBClassifier'
        else:
            self.operator_name = 'XGBRegressor'

    def get_xgb_params(self):
        return self.kwargs

    def get_booster(self):
        return self.booster_

    def _generate_classes(self, model_dict):
        if model_dict['num_class'] == 1:
            return np.asarray([0, 1])
        return np.arange(model_dict['num_class'])        


def _get_xgboost_operator_name(model):
    '''
    Get operator name of the input argument

    :param model_type:  A xgboost object.
    :return: A string which stands for the type of the input model in our conversion framework
    '''
    if isinstance(model, WrappedBooster):
        return model.operator_name
    if type(model) not in xgboost_operator_name_map:
        raise ValueError("No proper operator name found for '%s'" % type(model))
    return xgboost_operator_name_map[type(model)]


def _parse_xgboost_simple_model(scope, model, inputs):
    '''
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A xgboost object
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next stage
    '''
    this_operator = scope.declare_local_operator(_get_xgboost_operator_name(model), model)
    this_operator.inputs = inputs

    if (type(model) in xgboost_classifier_list or
            getattr(model, 'operator_name', None) == 'XGBClassifier'):
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

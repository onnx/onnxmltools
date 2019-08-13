# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy

from ..common._container import LightGbmModelContainer
from ..common._topology import *
from ..common.data_types import FloatTensorType

from lightgbm import LGBMClassifier, LGBMRegressor

lightgbm_classifier_list = [LGBMClassifier]

# Associate scikit-learn types with our operator names. If two scikit-learn models share a single name, it means their
# are equivalent in terms of conversion.
lightgbm_operator_name_map = {LGBMClassifier: 'LgbmClassifier',
                              LGBMRegressor: 'LgbmRegressor'}

class WrappedBooster:

    def __init__(self, booster):
        self.booster_ = booster
        _model_dict = self.booster_.dump_model()
        self.classes_ = self._generate_classes(_model_dict)
        self.n_features_ = len(_model_dict['feature_names'])
        if _model_dict['objective'].startswith('binary'):
            self.operator_name = 'LgbmClassifier'
        elif _model_dict['objective'].startswith('regression'):
            self.operator_name = 'LgbmRegressor'
        else:
            # Multiclass classifier is not supported at the moment. The exported ONNX model
            # has an issue in ZipMap node.
            raise ValueError('unsupported LightGbm objective: {}'.format(_model_dict['objective']))
        if _model_dict.get('average_output', False):
            self.boosting_type = 'rf'
        else:
            # Other than random forest, other boosting types do not affect later conversion.
            # Here `gbdt` is chosen for no reason.
            self.boosting_type = 'gbdt'

    def _generate_classes(self, model_dict):
        if model_dict['num_class'] == 1:
            return numpy.asarray([0, 1])
        return numpy.arange(model_dict['num_class'])


def _get_lightgbm_operator_name(model):
    '''
    Get operator name of the input argument

    :param model:  A lightgbm object.
    :return: A string which stands for the type of the input model in our conversion framework
    '''
    if isinstance(model, WrappedBooster):
        return model.operator_name
    model_type = type(model)
    if model_type not in lightgbm_operator_name_map:
        raise ValueError("No proper operator name found for '%s'" % model_type)
    return lightgbm_operator_name_map[model_type]


def _parse_lightgbm_simple_model(scope, model, inputs):
    '''
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A lightgbm object
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next stage
    '''
    operator_name = _get_lightgbm_operator_name(model)
    this_operator = scope.declare_local_operator(operator_name, model)
    this_operator.inputs = inputs

    if operator_name == 'LgbmClassifier':
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


def _parse_lightgbm(scope, model, inputs):
    '''
    This is a delegate function. It doesn't nothing but invoke the correct parsing function according to the input
    model's type.
    :param scope: Scope object
    :param model: A lightgbm object
    :param inputs: A list of variables
    :return: The output variables produced by the input model
    '''
    return _parse_lightgbm_simple_model(scope, model, inputs)


def parse_lightgbm(model, initial_types=None, target_opset=None,
                   custom_conversion_functions=None, custom_shape_calculators=None):

    raw_model_container = LightGbmModelContainer(model)
    topology = Topology(raw_model_container,
                        initial_types=initial_types, target_opset=target_opset,
                        custom_conversion_functions=custom_conversion_functions,
                        custom_shape_calculators=custom_shape_calculators)
    scope = topology.declare_scope('__root__')

    inputs = []
    for var_name, initial_type in initial_types:
        inputs.append(scope.declare_local_variable(var_name, initial_type))

    for variable in inputs:
        raw_model_container.add_input(variable)

    outputs = _parse_lightgbm(scope, model, inputs)

    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology

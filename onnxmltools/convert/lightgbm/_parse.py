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
        if (_model_dict['objective'].startswith('binary') or
                _model_dict['objective'].startswith('multiclass')):
            self.operator_name = 'LgbmClassifier'
        elif _model_dict['objective'].startswith('regression'):
            self.operator_name = 'LgbmRegressor'
        else:
            # Other objectives are not supported.
            raise ValueError("Unsupported LightGbm objective: '{}'.".format(_model_dict['objective']))
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


def _parse_sklearn_classifier(scope, model, inputs):
    probability_tensor = _parse_lightgbm_simple_model(
            scope, model, inputs)
    this_operator = scope.declare_local_operator('LgbmZipMap')
    this_operator.inputs = probability_tensor
    classes = model.classes_
    label_type = Int64Type()

    if (isinstance(model.classes_, list) and
            isinstance(model.classes_[0], numpy.ndarray)):
        # multi-label problem
        # this_operator.classlabels_int64s = list(range(0, len(classes)))
        raise NotImplementedError("multi-label is not supported")
    elif numpy.issubdtype(model.classes_.dtype, numpy.floating):
        classes = numpy.array(list(map(lambda x: int(x), classes)))
        if set(map(lambda x: float(x), classes)) != set(model.classes_):
            raise RuntimeError("skl2onnx implicitly converts float class "
                               "labels into integers but at least one label "
                               "is not an integer. Class labels should "
                               "be integers or strings.")
        this_operator.classlabels_int64s = classes
    elif numpy.issubdtype(model.classes_.dtype, numpy.signedinteger):
        this_operator.classlabels_int64s = classes
    else:
        classes = numpy.array([s.encode('utf-8') for s in classes])
        this_operator.classlabels_strings = classes
        label_type = StringType()

    output_label = scope.declare_local_variable('output_label', label_type)
    output_probability = scope.declare_local_variable(
        'output_probability',
        SequenceType(DictionaryType(label_type, FloatTensorType())))
    this_operator.outputs.append(output_label)
    this_operator.outputs.append(output_probability)
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
    if isinstance(model, LGBMClassifier):
        return _parse_sklearn_classifier(scope, model, inputs)
    if (isinstance(model, WrappedBooster) and
            model.operator_name == 'LgbmClassifier'):
        return _parse_sklearn_classifier(scope, model, inputs)
    return _parse_lightgbm_simple_model(scope, model, inputs)


def parse_lightgbm(model, initial_types=None, target_opset=None,
                   custom_conversion_functions=None, custom_shape_calculators=None):

    raw_model_container = LightGbmModelContainer(model)
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

    outputs = _parse_lightgbm(scope, model, inputs)

    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology

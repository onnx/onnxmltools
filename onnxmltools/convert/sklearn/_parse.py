# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._container import SklearnModelContainer
from ..common._topology import *

# Pipeline
from sklearn import pipeline

# Linear classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

# Linear regressors
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR

# Tree-based models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# Support vector machines
from sklearn.svm import SVC, SVR, NuSVC, NuSVR

# Operators for preprocessing and feature engineering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from lightgbm import LGBMClassifier, LGBMRegressor

# In most cases, scikit-learn operator produces only one output. However, each classifier has basically two outputs;
# one is the predicted label and the other one is the probabilities of all possible labels. Here is a list of supported
# scikit-learn classifiers. In the parsing stage, we produce two outputs for objects included in the following list and
# one output for everything not in the list.
sklearn_classifier_list = [LogisticRegression, SGDClassifier, LinearSVC, SVC, NuSVC,
                           GradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier,
                           ExtraTreesClassifier, LGBMClassifier]

# Associate scikit-learn types with our operator names. If two scikit-learn models share a single name, it means their
# are equivalent in terms of conversion.
sklearn_operator_name_map = {RobustScaler: 'SklearnRobustScaler',
                             StandardScaler: 'SklearnScaler',
                             LogisticRegression: 'SklearnLinearClassifier',
                             SGDClassifier: 'SklearnLinearClassifier',
                             LinearSVC: 'SklearnLinearSVC',
                             OneHotEncoder: 'SklearnOneHotEncoder',
                             DictVectorizer: 'SklearnDictVectorizer',
                             Imputer: 'SklearnImputer',
                             LabelEncoder: 'SklearnLabelEncoder',
                             SVC: 'SklearnSVC',
                             NuSVC: 'SklearnSVC',
                             SVR: 'SklearnSVR',
                             NuSVR: 'SklearnSVR',
                             LinearSVR: 'SklearnLinearSVR',
                             ElasticNet: 'SklearnElasticNetRegressor',
                             LinearRegression: 'SklearnLinearRegressor',
                             Ridge: 'SklearnLinearRegressor',
                             SGDRegressor: 'SklearnLinearRegressor',
                             Normalizer: 'SklearnNormalizer',
                             DecisionTreeClassifier: 'SklearnDecisionTreeClassifier',
                             DecisionTreeRegressor: 'SklearnDecisionTreeRegressor',
                             RandomForestClassifier: 'SklearnRandomForestClassifier',
                             RandomForestRegressor: 'SklearnRandomForestRegressor',
                             ExtraTreesClassifier: 'SklearnExtraTreesClassifier',
                             ExtraTreesRegressor: 'SklearnExtraTreesRegressor',
                             GradientBoostingClassifier: 'SklearnGradientBoostingClassifier',
                             GradientBoostingRegressor: 'SklearnGradientBoostingRegressor',
                             Binarizer: 'SklearnBinarizer',
                             LGBMClassifier: 'LgbmClassifier',
                             LGBMRegressor: 'LgbmRegressor',
                             TruncatedSVD: 'SklearnTruncatedSVD',
                             MinMaxScaler: 'SklearnMinMaxScaler',
                             MaxAbsScaler: 'SklearnMaxAbsScaler'}


def _get_sklearn_operator_name(model_type):
    '''
    Get operator name of the input argument

    :param model_type:  A scikit-learn object (e.g., SGDClassifier and Binarizer)
    :return: A string which stands for the type of the input model in our conversion framework
    '''
    if model_type not in sklearn_operator_name_map:
        print(sklearn_operator_name_map)
        raise ValueError('No proper operator name found for %s' % model_type)
    return sklearn_operator_name_map[model_type]


def _parse_sklearn_simple_model(scope, model, inputs):
    '''
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder and LogisticRegression)
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next stage
    '''
    print('simple model: %s ' % type(model))
    this_operator = scope.declare_local_operator(_get_sklearn_operator_name(type(model)), model)
    this_operator.inputs = inputs

    if type(model) in sklearn_classifier_list:
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


def _parse_sklearn_pipeline(scope, model, inputs):
    '''
    The basic ideas of scikit-learn parsing:
        1. Sequentially go though all stages defined in the considered scikit-learn pipeline
        2. The output variables of one stage will be fed into its next stage as the inputs.

    :param scope: Scope object defined in _topology.py
    :param model: scikit-learn pipeline object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by the input pipeline
    '''
    print('pipeline: %s ' % type(model))
    for step in model.steps:
        inputs = _parse_sklearn(scope, step[1], inputs)
    return inputs


def _parse_sklearn(scope, model, inputs):
    '''
    This is a delegate function. It doesn't nothing but invoke the correct parsing function according to the input
    model's type.
    :param scope: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder and LogisticRegression)
    :param inputs: A list of variables
    :return: The output variables produced by the input model
    '''
    if isinstance(model, pipeline.Pipeline):
        return _parse_sklearn_pipeline(scope, model, inputs)
    else:
        return _parse_sklearn_simple_model(scope, model, inputs)


def parse_sklearn(model, initial_types=None, targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    # Put scikit-learn object into an abstract container so that our framework can work seamlessly on models created
    # with different machine learning tools.
    raw_model_container = SklearnModelContainer(model)

    # Declare a computational graph. It will become a representation of the input scikit-learn model after parsing.
    topology = Topology(raw_model_container, initial_types=initial_types, targeted_onnx=targeted_onnx,
        custom_conversion_functions = custom_conversion_functions, custom_shape_calculators = custom_shape_calculators)

    # Declare an object to provide variables' and operators' naming mechanism. In contrast to CoreML, one global scope
    # is enough for parsing scikit-learn models.
    scope = topology.declare_scope('__root__')

    # Declare input variables. They should be the inputs of the scikit-learn model you want to convert into ONNX
    inputs = []
    for var_name, initial_type in initial_types:
        inputs.append(scope.declare_local_variable(var_name, initial_type))

    # The object raw_model_container is a part of the topology we're going to return. We use it to store the inputs of
    # the scikit-learn's computational graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input scikit-learn model as a Topology object.
    outputs = _parse_sklearn(scope, model, inputs)

    # THe object raw_model_container is a part of the topology we're going to return. We use it to store the outputs of
    # the scikit-learn's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology

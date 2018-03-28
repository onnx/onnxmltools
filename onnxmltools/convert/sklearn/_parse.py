import graphviz
from ..coreml._topology import *

# Pipeline
from sklearn import pipeline

# Linear classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

# Linear regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR

# Tree-based models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# Support vector machines
from sklearn.svm import SVC, SVR, NuSVC, NuSVR

# Operators for preprocessing and feature engineering
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

sklearn_classifier_list = [LogisticRegression, SGDClassifier, LinearSVC, SVC, NuSVC,
                           GradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier]

# Associate scikit-learn types with our operator names
sklearn_operator_name_map = {StandardScaler: 'SklearnScaler',
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
                             LinearRegression: 'SklearnLinearRegressor',
                             Ridge: 'SklearnLinearRegressor',
                             SGDRegressor: 'SklearnLinearRegressor',
                             Normalizer: 'SklearnNormalizer',
                             DecisionTreeClassifier: 'SklearnDecisionTreeClassifier',
                             DecisionTreeRegressor: 'SklearnDecisionTreeRegressor',
                             RandomForestClassifier: 'SklearnRandomForestClassifier',
                             RandomForestRegressor: 'SklearnRandomForestRegressor',
                             GradientBoostingClassifier: 'SklearnGradientBoostingClassifier',
                             GradientBoostingRegressor: 'SklearnGradientBoostingRegressor',
                             Binarizer: 'SklearnBinarizer'}


def _get_sklearn_operator_name(model_type):
    if model_type not in sklearn_operator_name_map:
        print(sklearn_operator_name_map)
        raise ValueError('No proper operator name found for %s' % model_type)
    return sklearn_operator_name_map[model_type]


def _parse_sklearn_simple_model(scope, model, inputs):
    '''
    :param scope: Scope object
    :param model: a scikit-learn object (e.g., OneHotEncoder and LogisticRegression)
    :param inputs: a list of variables
    :return: a list of output variables which will be passed to next stage
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
    :param model: scikit-learn pipeline object
    :param inputs: A list of Variable object
    :param topology:
    '''
    print('pipeline: %s ' % type(model))
    for step in model.steps:
        inputs = _parse_sklearn(scope, step[1], inputs)
    return inputs


def _parse_sklearn(scope, model, inputs):
    if isinstance(model, pipeline.Pipeline):
        return _parse_sklearn_pipeline(scope, model, inputs)
    else:
        return _parse_sklearn_simple_model(scope, model, inputs)


def parse_sklearn(model, initial_types=None):
    raw_model_container = SklearnModelContainer(model)
    topology = Topology(raw_model_container)
    scope = topology.declare_scope('__root__')

    # Declare input variables. They should be the inputs of the scikit-learn model you want to convert into ONNX.
    if not initial_types:
        raise ValueError('Initial types are required')
    inputs = []
    for i, initial_type in enumerate(initial_types):
        inputs.append(scope.declare_local_variable('input' + str(i), initial_type))

    for variable in inputs:
        raw_model_container.add_input(variable)
    outputs = _parse_sklearn(scope, model, inputs)
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology


def visualize_topology(topology, filename=None, view=True):
    graph = graphviz.Digraph()
    # declare nodes (variables and operators)
    for scope in topology.scopes:
        for variable in scope.variables.values():
            if variable.is_root:
                graph.attr('node', shape='oval', style='filled', fillcolor='blue')
            else:
                graph.attr('node', shape='oval', style='filled', fillcolor='white')
                if type(variable.type) != DictionaryType:
                    graph.node(variable.full_name, label=variable.full_name + ', ' + str(variable.type.shape))
                else:
                    graph.node(variable.full_name, label=variable.full_name + ', Dictionary')
    for operator in scope.operators.values():
        graph.attr('node', shape='box', style='filled', fillcolor='green')
        graph.node(operator.full_name, label=operator.full_name)
    # declare edges (connections between variables and operators)
    for scope in topology.scopes:
        for operator in scope.operators.values():
            for variable in operator.inputs:
                graph.edge(variable.full_name, operator.full_name)
            for variable in operator.outputs:
                graph.edge(operator.full_name, variable.full_name)
    if filename is not None:
        graph.render(filename, view=view)
    return graph

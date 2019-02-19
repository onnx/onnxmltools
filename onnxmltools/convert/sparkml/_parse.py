# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._container import SparkmlModelContainer
from ..common._topology import *

from pyspark.ml import PipelineModel


from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import ChiSqSelectorModel
from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.feature import DCT
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDFModel
from pyspark.ml.feature import ImputerModel
from pyspark.ml.feature import IndexToString
from pyspark.ml.feature import MaxAbsScalerModel
from pyspark.ml.feature import MinHashLSHModel
from pyspark.ml.feature import MinMaxScalerModel
from pyspark.ml.feature import NGram
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import OneHotEncoderModel
from pyspark.ml.feature import PCAModel
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import RFormulaModel
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.feature import StandardScalerModel
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexerModel
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import Word2VecModel

from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayesModel
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import OneVsRestModel

from pyspark.ml.regression import AFTSurvivalRegressionModel
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.regression import GeneralizedLinearRegressionModel
from pyspark.ml.regression import IsotonicRegressionModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressor

from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import LDA

# In most cases, spark-ml operator produces only one output. However, each classifier has basically two outputs:
#   1. prediction: selected label
#   2. rawPrediction: Dense vector containing values for each class
# Here is a list of supported spark-ml classifiers.
# In the parsing stage, we produce two outputs for objects included in the following list and
# one output for everything not in the list.
sparkml_classifier_list = [LinearSVCModel, LogisticRegressionModel, DecisionTreeClassifier, GBTClassifier,
                           RandomForestClassifier, NaiveBayesModel, MultilayerPerceptronClassifier, OneVsRestModel]

# Associate spark-ml types with our operator names. If two spark-ml models share a single name, it means their
# are equivalent in terms of conversion.

def build_sparkml_operator_name_map():
    res = {k: "pyspark.ml.feature." + k.__name__ for k in [
        Binarizer, BucketedRandomProjectionLSHModel, Bucketizer,
        ChiSqSelectorModel, CountVectorizerModel, DCT, ElementwiseProduct, HashingTF, IDFModel, ImputerModel,
        IndexToString, MaxAbsScalerModel, MinHashLSHModel, MinMaxScalerModel, NGram, Normalizer, OneHotEncoderModel,
        PCAModel, PolynomialExpansion, QuantileDiscretizer, RegexTokenizer, RFormulaModel, SQLTransformer,
        StandardScalerModel, StopWordsRemover, StringIndexerModel, Tokenizer, VectorAssembler, VectorIndexerModel,
        VectorSlicer, Word2VecModel
    ]}
    res.update({k: "pyspark.ml.classification." + k.__name__ for k in [
        LinearSVCModel, LogisticRegressionModel, DecisionTreeClassifier, GBTClassifier, RandomForestClassifier,
        NaiveBayesModel, MultilayerPerceptronClassifier, OneVsRestModel
    ]})
    res.update({k: "pyspark.ml.regression." + k.__name__ for k in [
        AFTSurvivalRegressionModel, DecisionTreeRegressor, GBTRegressionModel, GBTRegressionModel,
        GeneralizedLinearRegressionModel, IsotonicRegressionModel, LinearRegressionModel, RandomForestRegressor
    ]})
    return res


sparkml_operator_name_map = build_sparkml_operator_name_map()

def build_io_name_map():
    map = {
        "pyspark.ml.feature.Normalizer": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")]
        ),
        "pyspark.ml.feature.Binarizer": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")]
        ),
        "pyspark.ml.classification.LogisticRegressionModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol"), model.getOrDefault("probabilityCol")]
        ),
        "pyspark.ml.feature.OneHotEncoderModel": (
            lambda model: model.getOrDefault("inputCols"),
            lambda model: model.getOrDefault("outputCols")
        ),
        "pyspark.ml.feature.StringIndexerModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")]
        ),
        "pyspark.ml.feature.VectorAssembler": (
            lambda model: model.getOrDefault("inputCols"),
            lambda model: [model.getOrDefault("outputCol")]
        )
    }
    return map

io_name_map = build_io_name_map()

def _get_input_names(model):
    '''
    Returns the name(s) of the input(s) for a SparkML operator
    :param model: SparkML Model
    :return: list of input names
    '''
    return io_name_map[_get_sparkml_operator_name(type(model))][0](model)


def _get_output_names(model):
    '''
    Returns the name(s) of the output(s) for a SparkML operator
    :param model: SparkML Model
    :return: list of output names
    '''
    return io_name_map[_get_sparkml_operator_name(type(model))][1](model)


def _get_sparkml_operator_name(model_type):
    '''
    Get operator name of the input argument

    :param model_type:  A spark-ml object (LinearRegression, StringIndexer, ...)
    :return: A string which stands for the type of the input model in our conversion framework
    '''
    if model_type not in sparkml_operator_name_map:
        raise ValueError("No proper operator name found for '%s'" % model_type)
    return sparkml_operator_name_map[model_type]


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

def _parse_sparkml_simple_model(scope, model, global_inputs, output_dict):
    '''
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A spark-ml Transformer/Evaluator (e.g., OneHotEncoder and LogisticRegression)
    :param global_inputs: A list of variables
    :param output_dict: An accumulated list of output_original_name->(ref_count, variable)
    :return: A list of output variables which will be passed to next stage
    '''
    this_operator = scope.declare_local_operator(_get_sparkml_operator_name(type(model)), model)
    raw_input_names = _get_input_names(model)
    this_operator.inputs = [_get_variable_for_input(scope, x, global_inputs, output_dict) for x in raw_input_names]
    raw_output_names = _get_output_names(model)
    for output_name in raw_output_names:
        variable = scope.declare_local_variable(output_name, FloatTensorType())
        this_operator.outputs.append(variable)
        output_dict[variable.raw_name] = [0, variable]


    # if type(model) in sparkml_classifier_list:
    #     # For classifiers, we may have two outputs, one for label and the other one for probabilities of all classes.
    #     # Notice that their types here are not necessarily correct and they will be fixed in shape inference phase
    #     label_variable = scope.declare_local_variable('label', FloatTensorType())
    #     probability_map_variable = scope.declare_local_variable('probabilities', FloatTensorType())
    #     this_operator.outputs.append(label_variable)
    #     this_operator.outputs.append(probability_map_variable)
    #     output_dict[label_variable.raw_name] = [0, label_variable]
    #     output_dict[probability_map_variable.raw_name] = [0, probability_map_variable]
    # else:
    #     # We assume that all spark-ml operator can only produce a single float tensor.
    #     variable = scope.declare_local_variable('output', FloatTensorType())
    #     this_operator.outputs.append(variable)
    #     output_dict[variable.raw_name] = [0, variable]


def _parse_sparkml_pipeline(scope, model, global_inputs, output_dict):
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
        _parse_sparkml(scope, stage, global_inputs, output_dict)

def _parse_sparkml(scope, model, global_inputs, output_dict):
    '''
    This is a delegate function. It doesn't nothing but invoke the correct parsing function according to the input
    model's type.
    :param scope: Scope object
    :param model: A spark-ml object (e.g., OneHotEncoder and LogisticRegression)
    :param inputs: A list of variables
    :return: The output variables produced by the input model
    '''
    if isinstance(model, PipelineModel):
        return _parse_sparkml_pipeline(scope, model, global_inputs, output_dict)
    else:
        return _parse_sparkml_simple_model(scope, model, global_inputs, output_dict)


def parse_sparkml(model, initial_types=None, target_opset=None,
                  custom_conversion_functions=None, custom_shape_calculators=None):
    # Put spark-ml object into an abstract container so that our framework can work seamlessly on models created
    # with different machine learning tools.
    raw_model_container = SparkmlModelContainer(model)

    # Declare a computational graph. It will become a representation of the input spark-ml model after parsing.
    topology = Topology(raw_model_container,
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
    _parse_sparkml(scope, model, inputs, output_dict)
    outputs = []
    for k, v in output_dict.items():
        if v[0] == 0: # ref count is zero
            outputs.append(v[1])

    # THe object raw_model_container is a part of the topology we're going to return. We use it to store the outputs of
    # the spark-ml's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology

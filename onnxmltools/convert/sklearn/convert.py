#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import copy
import sklearn.pipeline as pipeline
from ..common import ModelBuilder
from ..common import registration
from .SklearnConvertContext import SklearnConvertContext as ConvertContext
from .datatype import convert_incoming_type

# These are not referenced directly but are imported to initialize the registration call
from .TreeEnsembleConverter import *
from .BinarizerConverter import BinarizerConverter
from .DictVectorizerConverter import DictVectorizerConverter
from .ImputerConverter import ImputerConverter
from .LabelEncoderConverter import LabelEncoderConverter
from .NormalizerConverter import NormalizerConverter
from .OneHotEncoderConverter import OneHotEncoderConverter
from .ScalerConverter import ScalerConverter
from .TreeEnsembleConverter import DecisionTreeClassifierConverter
from .TreeEnsembleConverter import DecisionTreeRegressorConverter
from .TreeEnsembleConverter import RandomForestClassifierConverter
from .TreeEnsembleConverter import RandomForestRegressorConverter
from .TreeEnsembleConverter import GradientBoostingClassifierConverter
from .TreeEnsembleConverter import GradientBoostingRegressorConverter
from .GLMClassifierConverter import GLMClassifierConverter
from .GLMRegressorConverter import GLMRegressorConverter
from .SVMConverter import SVCConverter, SVMConverter


def convert(model, name=None, input_features=None):
    '''
    This function produces an equivalent ONNX model of the given scikit-learn model. The supported scikit-learn
    modules are listed below.
    * Preprocessings and transformations:
      1.  feature_extraction.DictVectorizer
      2.  preprocessing.Imputer
      3.  preprocessing.LabelEncoder
      4.  preprocessing.Normalizer
      5.  preprocessing.OneHotEncoder
      6.  preprocessing.StandardScaler
    * Linear classification and regression:
      7.  svm.LinearSVC
      8.  linear_model.LogisticRegression,
      9.  linear_model.SGDClassifier
      10. svm.LinearSVR
      11. linear_model.LinearRegression
      12. linear_model.Ridge
      13. linear_model.SGDRegressor
    * Support vector machine for classification and regression
      14. svm.SVC
      15. svm.SVR
      16. svm.NuSVC
      17. svm.NuSVR
    * Tree-based models for classification and regression
      18. tree.DecisionTreeClassifier
      19. tree.DecisionTreeRegressor
      20. ensemble.GradientBoostingClassifier
      21. ensemble.GradientBoostingRegressor
      22. ensemble.RandomForestClassifier
      23. ensemble.RandomForestRegressor
    * pipeline
      24. pipeline.Pipeline
    For pipeline conversion, user needs to make sure each component is one of our supported items (1)-(24).
    :param model: A scikit-learn object form the list above
    :param name: Name of the ONNX graph (type: GraphProto) in the generated ONNX model (type: ModelProto)
    :param input_features: A list of tuples specifying input feature types. The tuple format is (feature_name,
    input_type, input_shape), where feature_name is a string. There are several major input types.
      1. Tensors: Its input_type (a string) is one of tensor element types allowed in ONNX. Common values of input_type
      include "float," "int64," and "string." User can specify the tensor shape using input_shape, which is an integer
      or a list of integers.
      2. Dictionary: Its input_type is a string with format "DictKeyType_ValueType," where each of KeyType and ValueType
      is one of ONNX's tensor element types. For a dictionary with string keys and float values, we can use
      "input_type=Dictstring_float." Dimension is not needed so input_shape would be ignored if specified.
    :return: An equivalent ONNX model (type: ModelProto) of the input scikit-learn model
    '''
    context = ConvertContext()

    # model_inputs are the inputs to the model
    # node_inputs are the current set of inputs that are going into the converted node
    model_inputs = []
    node_inputs = []
    nodes = []
    if input_features:
        try:
            for input_name, input_type, input_shape in input_features:
                # Register the input name
                context.get_unique_name(input_name)
                converted_input = convert_incoming_type(input_name, input_type, input_shape)
                model_inputs.append(converted_input)

            # Merge the inputs into a feature vectorizer
            if len(input_features) > 1:
                nodes.extend(_combine_inputs(context, model_inputs))
                node_inputs = nodes[-1].outputs
            else:
                # Set node_inputs to be the model inputs since there are no features to override.
                node_inputs = copy.deepcopy(model_inputs)
        except:
            raise ValueError('Invalid input_features argument.')
    else:
        # Use the default input argument
        node_inputs = [model_util.make_tensor_value_info('Input', onnx_proto.TensorProto.FLOAT)]
        model_inputs = copy.deepcopy(node_inputs)

    nodes += _convert_sklearn_node(context, model, node_inputs)
    mb = ModelBuilder(name)
    for node in nodes:
        mb.add_nodes([node.onnx_node])
        mb.add_initializers(node.initializers)
        mb.add_values(node.values)
        mb.add_op_set(node.op_set)

    mb.add_inputs(model_inputs)

    # The last converter could have set outputs to use for the model.
    # Therefore use these outputs if set, otherwise use the outputs from the last
    # node
    if len(context.outputs) > 0:
        mb.add_outputs(context.outputs)
    else:
        mb.add_outputs(nodes[-1].outputs)
    return mb.make_model()


def _convert_sklearn_pipeline(context, pipeline, inputs):
    nodes = []
    for sk_node in pipeline.steps:
        context.clear_outputs()
        converted_nodes = _convert_sklearn_node(context, sk_node[1], inputs)
        nodes.extend(converted_nodes)

        # Grab the outputs from the context if set.
        # If not set, then use the outputs from the last node.
        if len(context.outputs) > 0:
            inputs = context.outputs
        else:
            inputs = converted_nodes[-1].outputs

    return nodes


def _do_convert(context, converter, sk_node, inputs):
    converter.validate(sk_node)
    node = converter.convert(context, sk_node, inputs)
    if isinstance(node, list):
        return node
    return [node]


def _convert_sklearn_node(context, node, inputs):
    if isinstance(node, pipeline.Pipeline):
        return _convert_sklearn_pipeline(context, node, inputs)
    else:
        converter = registration.get_converter(type(node))
        return _do_convert(context, converter, node, inputs)


def _combine_inputs(context, inputs):
    nodes = []

    # This combines all the inputs into a single input. This currently only handles ints and floats.
    # Ints are converted to a float by adding a Scalar operator into the graph
    # Strings are not supported and will assert if specified.

    onnx_unsupported_types = [onnx_proto.TensorProto.STRING]

    fv_inputs = []
    # Get inputs that are integer types
    for inp in inputs:
        if inp.type.tensor_type.elem_type in onnx_unsupported_types:
            raise RuntimeError("Unsupported type specified."
                               "These types are not supported when specifying multiple inputs {0}".format(onnx_unsupported_types))

        if inp.type.tensor_type.elem_type in model_util.onnx_integer_types:
            scale_node = model_util.create_scaler(inp, inp.name, 1.0, 0.0, context)
            nodes.append(scale_node)
            fv_inputs.append(scale_node.outputs[0])
        else:
            fv_inputs.append(inp)

    feature_vector = model_util.create_feature_vector(fv_inputs, 'input_vector', context)
    nodes.append(feature_vector)

    return nodes


# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from uuid import uuid4
from ..common._topology import convert_topology
from ._parse import parse_sklearn

# Invoke the registration of all our converters and shape calculators
from . import shape_calculators
from . import operator_converters


# The conversion of a scikit-learn model consists of several major steps.
#   1. Put the input model into a Container object (i.e., onnxmltools.convert.common._container.SklearnModelContainer)
#   2. Create a empty Topology (i.e., onnxmltools.convert.common._topology.Topology) object. It will be an an abstract
#      computational graph of the input model.
#   3. Parse the scikit-learn model as a computational graph. We may add variables and operators into the Topology
#      object defined in the previous step.
#   4. Call the member function, compile, in Topology. There are two important steps.
#        a. First, we may feed input variables specified in the Container (defined in Step 1) into our computational
#           graph and then evaluate the computational graph. Unreachable variables and operators would be removed.
#        b. Second, we may invoke the shape calculators for all existing operators in a topological order.
#   5. Finally, the conversion functions of all existing operators are called in a topological order.
#
#   Note that steps 1-3 can be found in onnxmltools.convert.sklearn._parse.parse_sklearn. Step 5 is implemented in
#   onnxmltools.common._topology.convert_topology. Step 4-a is onnxmltools.convert.common._topology.Topology._prune and
#   step 4-b is onnxmltools.convert.common._topology.Topology._infer_all_types.
def convert(model, name=None, initial_types=None, doc_string=''):
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

    This function converts the specified scikit-learn model into its ONNX counterpart. Notice that for all conversions,
    initial types are required.  ONNX model name can also be specified.

    :param model: A scikit-learn model
    :param initial_types: a python list whose elements are data types defined in data_types.py
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :return: An ONNX model (type: ModelProto) which is equivalent to the input scikit-learn model

    Example of initial types:
    Assume that the specified scikit-learn model takes a heterogeneous list as its input. If the first 5 elements are
    floats and the last 10 elements are integers, we need to specify initial types as below. The [1] in [1, 5] indicates
    the batch size here is 1.
    >>> from onnxmltools.convert.common.data_types import FloatTensorType
    >>> initial_type = [FloatTensorType([1, 5]), Int64TensorType([1, 10])]
    '''
    if initial_types is None:
        raise ValueError('Initial types are required')

    if name is None:
        name = str(uuid4().hex)

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    topology = parse_sklearn(model, initial_types)

    # Infer variable shapes
    topology.compile()

    # Convert our Topology object into ONNX. The outcome is an ONNX model.
    onnx_model = convert_topology(topology, name, doc_string)

    return onnx_model

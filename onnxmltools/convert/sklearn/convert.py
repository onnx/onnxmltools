# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from uuid import uuid4
from ...proto import onnx
from ..common._topology import convert_topology
from ._parse import parse_sklearn

# Invoke the registration of all our converters and shape calculators
from . import shape_calculators
from . import operator_converters


def convert(model, name=None, initial_types=None, doc_string='',
            targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    '''
    This function produces an equivalent ONNX model of the given scikit-learn model. The supported scikit-learn
    modules are listed below.

    * Preprocessings and transformations:
      1.  feature_extraction.DictVectorizer
      2.  preprocessing.Imputer
      3.  preprocessing.LabelEncoder
      4.  preprocessing.Normalizer
      5.  preprocessing.OneHotEncoder
      6.  preprocessing.RobustScale
      7.  preprocessing.StandardScaler
      8.  decomposition.TruncatedSVD
    * Linear classification and regression:
      9.  svm.LinearSVC
      10. linear_model.LogisticRegression,
      11. linear_model.SGDClassifier
      12. svm.LinearSVR
      13. linear_model.LinearRegression
      14. linear_model.Ridge
      15. linear_model.SGDRegressor
      16. linear_model.ElasticNet
    * Support vector machine for classification and regression
      17. svm.SVC
      18. svm.SVR
      19. svm.NuSVC
      20. svm.NuSVR
    * Tree-based models for classification and regression
      21. tree.DecisionTreeClassifier
      22. tree.DecisionTreeRegressor
      23. ensemble.GradientBoostingClassifier
      24. ensemble.GradientBoostingRegressor
      25. ensemble.RandomForestClassifier
      26. ensemble.RandomForestRegressor
      27. ensemble.ExtraTreesClassifier
      28. ensemble.ExtraTreesRegressor
    * LightGBM Python module
      29. LGBMClassifiers (http://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMClassifier)
      30. LGBMRegressor (http://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMRegressor)
    * pipeline
      31. pipeline.Pipeline

    For pipeline conversion, user needs to make sure each component is one of our supported items (1)-(24).

    This function converts the specified scikit-learn model into its ONNX counterpart. Notice that for all conversions,
    initial types are required.  ONNX model name can also be specified.

    :param model: A scikit-learn model
    :param initial_types: a python list. Each element is a tuple of a variable name and a type defined in data_types.py
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :param targeted_onnx: A string (for example, '1.1.2' and '1.2') used to specify the targeted ONNX version of the
    produced model. If ONNXMLTools cannot find a compatible ONNX python package, an error may be thrown.
    :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
    :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
    :return: An ONNX model (type: ModelProto) which is equivalent to the input scikit-learn model

    Example of initial_types:
    Assume that the specified scikit-learn model takes a heterogeneous list as its input. If the first 5 elements are
    floats and the last 10 elements are integers, we need to specify initial types as below. The [1] in [1, 5] indicates
    the batch size here is 1.
    >>> from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType
    >>> initial_type = [('float_input', FloatTensorType([1, 5])), ('int64_input', Int64TensorType([1, 10]))]
    '''
    if initial_types is None:
        raise ValueError('Initial types are required. See usage of convert(...) in \
                         onnxmltools.convert.sklearn.convert for details')

    if name is None:
        name = str(uuid4().hex)

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    topology = parse_sklearn(model, initial_types, targeted_onnx, custom_conversion_functions, custom_shape_calculators)

    # Infer variable shapes
    topology.compile()

    # Convert our Topology object into ONNX. The outcome is an ONNX model.
    onnx_model = convert_topology(topology, name, doc_string, targeted_onnx)

    return onnx_model

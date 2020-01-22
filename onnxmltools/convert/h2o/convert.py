# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from uuid import uuid4
import json
import tempfile
import h2o

from ...proto import onnx, get_opset_number_from_onnx
from ..common._topology import convert_topology
from ..common.data_types import FloatTensorType
from ._parse import parse_h2o

# Invoke the registration of all our converters and shape calculators
from . import operator_converters, shape_calculators


def convert(model, name=None, initial_types=None, doc_string='', target_opset=None,
            targeted_onnx=onnx.__version__, custom_conversion_functions=None,
            custom_shape_calculators=None):
    '''
    This function produces an equivalent ONNX model of the given H2O MOJO model.
    Supported model types:
    - GBM, with limitations:
        - poisson, gamma, tweedie distributions not supported
        - multinomial distribution supported with 3 or more classes (use binomial otherwise)
    Ohter limitations:
    - modes with categorical splits not supported


    :param model: H2O MOJO model loaded into memory (see below for example)
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param initial_types: a python list. Each element is a tuple of a variable name and a type defined in data_types.py
    :param doc_string: A string attached onto the produced ONNX model
    :param target_opset: number, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.
    :param targeted_onnx: A string (for example, '1.1.2' and '1.2') used to specify the targeted ONNX version of the
        produced model. If ONNXMLTools cannot find a compatible ONNX python package, an error may be thrown.
    :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
    :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
    :return: An ONNX model (type: ModelProto) which is equivalent to the input xgboost model

    :examples:

    >>> from onnxmltools.convert import convert_h2o
    >>> file = open("/path/to/h2o_mojo.zip", "rb")
    >>> mojo_content = file.read()
    >>> file.close()
    >>> h2o_onnx_model = convert_h2o(mojo_content)
    '''
    if name is None:
        name = str(uuid4().hex)
    if initial_types is None:
        initial_types = [('input', FloatTensorType(shape=['None', 'None']))]

    _, model_path = tempfile.mkstemp()
    f = open(model_path, "wb")
    f.write(model)
    f.close()
    mojo_str = h2o.print_mojo(model_path, format="json")
    mojo_model = json.loads(mojo_str)
    if mojo_model["params"]["algo"] != "gbm":
        raise ValueError("Model type not supported (algo=%s). Only GBM Mojo supported for now." % mojo_model["params"]["algo"])

    target_opset = target_opset if target_opset else get_opset_number_from_onnx()
    topology = parse_h2o(mojo_model, initial_types, target_opset, custom_conversion_functions, custom_shape_calculators)
    topology.compile()
    onnx_model = convert_topology(topology, name, doc_string, target_opset, targeted_onnx)
    return onnx_model

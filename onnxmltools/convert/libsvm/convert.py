# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
import onnx
from ..common._topology import convert_topology
from ._parse import parse_libsvm

# Invoke the registration of all our converters and shape calculators
from . import shape_calculators
from . import operator_converters


def convert(model, name=None, initial_types=None, doc_string='', target_opset=None,
            targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    """
    :param model: a libsvm model
    :param initial_types: a python list. Each element is a tuple of a variable name and a type defined in data_types.py
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :param target_opset: number, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.
    :param targeted_onnx: A string (for example, '1.1.2' and '1.2') used to specify the targeted ONNX version of the
    produced model. If ONNXMLTools cannot find a compatible ONNX python package, an error may be thrown.
    :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
    :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
    :return: An ONNX model (type: ModelProto) which is equivalent to the input scikit-learn model
    """
    if initial_types is None:
        raise ValueError('Initial types are required. See usage of convert(...) in \
                         onnxmltools.convert.libsvm.convert for details')

    if name is None:
        name = str(uuid4().hex)
    target_opset = target_opset if target_opset else get_maximum_opset_supported()

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    topology = parse_libsvm(model, initial_types, custom_conversion_functions,
                            custom_shape_calculators)

    # Infer variable shapes
    topology.compile()

    # Convert our Topology object into ONNX. The outcome is an ONNX model.
    onnx_model = convert_topology(topology, name, doc_string, target_opset, targeted_onnx)

    return onnx_model

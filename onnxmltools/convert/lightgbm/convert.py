# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4
import onnx
import lightgbm
from ..common.onnx_ex import get_maximum_opset_supported
from ..common._topology import convert_topology
from ..common.utils import hummingbird_installed
from ._parse import parse_lightgbm, WrappedBooster

# Invoke the registration of all our converters and shape calculators
from . import operator_converters, shape_calculators  # noqa


def convert(
    model,
    name=None,
    initial_types=None,
    doc_string="",
    target_opset=None,
    targeted_onnx=onnx.__version__,
    custom_conversion_functions=None,
    custom_shape_calculators=None,
    without_onnx_ml=False,
    zipmap=True,
    split=None,
):
    """
    This function produces an equivalent ONNX model of the given lightgbm model.
    The supported lightgbm modules are listed below.

    * `LGBMClassifiers
      <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>`_
    * `LGBMRegressor
      <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_
    * `Booster
      <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html>`_

    :param model: A LightGBM model
    :param initial_types: a python list. Each element is a tuple
        of a variable name and a type defined in data_types.py
    :param name: The name of the graph (type: GraphProto) in the
        produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :param target_opset: number, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.
    :param targeted_onnx: A string (for example, '1.1.2' and '1.2')
        used to specify the targeted ONNX version of the
        produced model. If ONNXMLTools cannot find a compatible ONNX
            python package, an error may be thrown.
    :param custom_conversion_functions: a dictionary for specifying
        the user customized conversion function
    :param custom_shape_calculators: a dictionary for specifying
        the user customized shape calculator
    :param without_onnx_ml: whether to generate a model composed
        by ONNX operators only, or to allow the converter
    :param zipmap: remove operator ZipMap from the ONNX graph
    :param split: this parameter is usefull to reduce the level of discrepancies for
        big regression forest (number of trees > 100). lightgbm does all the computation
        with double whereas ONNX is using floats. Instead of having one single node
        TreeEnsembleRegressor, the converter splits it into
        multiple nodes TreeEnsembleRegressor,
        casts the output in double and before additioning all the outputs.
        The final graph is slower but keeps the discrepancies constant
        (it is proportional to the number of trees in a node TreeEnsembleRegressor).
        Parameter *split* is the number of trees per node. It could be possible to
        do the same with TreeEnsembleClassifier. However, the normalization of the
        probabilities significantly reduces the discrepancies.
    to use ONNX-ML operators as well.
    :return: An ONNX model (type: ModelProto) which is equivalent to the input lightgbm model
    """
    if initial_types is None:
        raise ValueError(
            "Initial types are required. See usage of convert(...) in "
            "onnxmltools.convert.lightgbm.convert for details"
        )
    if without_onnx_ml and not hummingbird_installed():
        raise RuntimeError(
            "Hummingbird is not installed. Please install hummingbird to use this feature: "
            "pip install hummingbird-ml"
        )
    if isinstance(model, lightgbm.Booster):
        model = WrappedBooster(model)
    if name is None:
        name = str(uuid4().hex)

    target_opset = target_opset if target_opset else get_maximum_opset_supported()
    topology = parse_lightgbm(
        model,
        initial_types,
        target_opset,
        custom_conversion_functions,
        custom_shape_calculators,
        zipmap=zipmap,
        split=split,
    )
    topology.compile()
    onnx_ml_model = convert_topology(
        topology, name, doc_string, target_opset, targeted_onnx
    )

    if without_onnx_ml:
        if zipmap:
            raise NotImplementedError(
                "Conversion with zipmap operator is not implemented with hummingbird-ml."
            )
        from hummingbird.ml import convert, constants

        extra_config = {}
        # extra_config[constants.ONNX_INITIAL_TYPES] = initial_types
        extra_config[constants.ONNX_OUTPUT_MODEL_NAME] = name
        extra_config[constants.ONNX_TARGET_OPSET] = target_opset
        onnx_model = convert(onnx_ml_model, "onnx", extra_config=extra_config).model
        return onnx_model

    return onnx_ml_model

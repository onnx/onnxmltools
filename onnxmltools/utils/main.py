# SPDX-License-Identifier: Apache-2.0

import os
from ..proto import onnx_proto
from ..convert.common import utils as convert_utils
from os import path


def load_model(file_path):
    """
    Loads an ONNX model to a ProtoBuf object.

    :param file_path: ONNX file (full file name)
    :return: ONNX model.

    Example:

    ::

        from onnxmltools.utils import load_model
        onnx_model = load_model("SqueezeNet.onnx")
    """
    if not path.exists(file_path):
        raise FileNotFoundError("{0} was not found.".format(file_path))
    model = onnx_proto.ModelProto()
    with open(file_path, "rb") as f:
        model.ParseFromString(f.read())
    return model


def save_model(model, file_path):
    """
    Saves an ONNX model to a ProtoBuf object.
    :param model: ONNX model
    :param file_path: ONNX file (full file name)

    Example:

    ::

        from onnxmltools.utils import save_model
        save_model(onnx_model, 'c:/test_model.onnx')
    """
    if model is None or not isinstance(model, onnx_proto.ModelProto):
        raise ValueError("Model is not a valid ONNX model.")
    directory = os.path.dirname(os.path.abspath(file_path))
    if not path.exists(directory):
        raise FileNotFoundError("Directory does not exist {0}".format(directory))
    with open(file_path, "wb") as f:
        f.write(model.SerializeToString())


def set_model_domain(model, domain):
    """
    Sets the domain on the ONNX model.

    :param model: instance of an ONNX model
    :param domain: string containing the domain name of the model

    Example:

    ::
        from onnxmltools.utils import set_model_domain
        onnx_model = load_model("SqueezeNet.onnx")
        set_model_domain(onnx_model, "com.acme")
    """
    if model is None or not isinstance(model, onnx_proto.ModelProto):
        raise ValueError("Model is not a valid ONNX model.")
    if not convert_utils.is_string_type(domain):
        raise ValueError("Domain must be a string type.")
    model.domain = domain


def set_model_version(model, version):
    """
    Sets the version of the ONNX model.

    :param model: instance of an ONNX model
    :param version: integer containing the version of the model

    Example:

    ::
        from onnxmltools.utils import set_model_version
        onnx_model = load_model("SqueezeNet.onnx")
        set_model_version(onnx_model, 1)
    """
    if model is None or not isinstance(model, onnx_proto.ModelProto):
        raise ValueError("Model is not a valid ONNX model.")
    if not convert_utils.is_numeric_type(version):
        raise ValueError("Version must be a numeric type.")
    model.model_version = version


def set_model_doc_string(model, doc, override=False):
    """
    Sets the doc string of the ONNX model.

    :param model: instance of an ONNX model
    :param doc: string containing the doc string that describes the model.
    :param override: bool if true will always override the doc string with the new value

    Example:

    ::
        from onnxmltools.utils import set_model_doc_string
        onnx_model = load_model("SqueezeNet.onnx")
        set_model_doc_string(onnx_model, "Sample doc string")
    """
    if model is None or not isinstance(model, onnx_proto.ModelProto):
        raise ValueError("Model is not a valid ONNX model.")
    if not convert_utils.is_string_type(doc):
        raise ValueError("Doc must be a string type.")
    if model.doc_string and not doc and override is False:
        raise ValueError(
            "Failing to overwrite the doc string with a "
            "blank string, set override to True if intentional."
        )
    model.doc_string = doc

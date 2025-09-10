# SPDX-License-Identifier: Apache-2.0

from .main import load_model
from .main import save_model
from .main import set_model_version
from .main import set_model_domain
from .main import set_model_doc_string
from .visualize import visualize_model

# from .float16_converter import convert_float_to_float16
from .tests_helper import dump_data_and_model
from .tests_helper import (
    dump_one_class_classification,
    dump_binary_classification,
    dump_multiple_classification,
)
from .tests_helper import dump_multiple_regression, dump_single_regression
from .tests_dl_helper import create_tensor


def convert_float_to_float16(*_args, **_kwargs):
    raise NotImplementedError(
        "Please use from onnxconverter_common.float16 import convert_float_to_float16."
    )

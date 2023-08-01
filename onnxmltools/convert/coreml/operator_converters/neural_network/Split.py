# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import apply_split
from ....common._registration import register_converter


def convert_split(scope, operator, container):
    # ONNX Split may evenly divide the input along the specified axis if "split" attribute is not specified.
    # Also, CoreML always evenly split the input along C-axis. Consequently, we only need to specify the axis
    # and make sure the number of outputs in ONNX matches that in CoreML.
    apply_split(scope, operator.input_full_names, operator.output_full_names, container,
                operator_name=operator.full_name, axis=1)


register_converter('split', convert_split)

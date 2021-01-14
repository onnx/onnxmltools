# SPDX-License-Identifier: Apache-2.0

import numpy as np
from .....proto import onnx_proto
from ....common._apply_operation import apply_crop_height_width
from ....common._registration import register_converter

def convert_crop(scope, operator, container):
    # Extract number of pixels cropped in CoreML operator.
    border = operator.raw_operator.crop.cropAmounts.borderAmounts

    # Compute cropping amounts. left_border=1 means one pixel will be removed at the beginning
    # of W-axis. bottom_border=1 means one pixel will be removed at the end of H-axis.
    left_border, top_border, right_border, bottom_border = (0,) * 4
    if len(operator.input_full_names) == 1:
        left_border = border[1].startEdgeSize
        top_border = border[0].startEdgeSize
        right_border = border[1].endEdgeSize
        bottom_border = border[0].endEdgeSize
    else:
        offset = operator.raw_operator.crop.offset
        in_shape = operator.inputs[0].type.shape
        out_shape = operator.outputs[0].type.shape
        left_border = offset[1]
        top_border = offset[0]
        right_border = in_shape[3] - left_border - out_shape[3]
        bottom_border = in_shape[2] - top_border - out_shape[2]

    # Delegate the selection of ONNX operator to a version-dependent function.
    apply_crop_height_width(scope,
        operator.input_full_names[0], operator.output_full_names[0],
        container, operator_name=operator.full_name,
        top_border=top_border, bottom_border=bottom_border,
        left_border=left_border, right_border=right_border)



register_converter('crop', convert_crop)

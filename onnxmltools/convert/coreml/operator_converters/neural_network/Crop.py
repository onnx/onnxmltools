# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_converter


def convert_crop(scope, operator, container):
    if len(operator.input_full_names) > 2:
        raise RuntimeError('Unlike CoreML, ONNX only supports cropping with a single input')

    op_type = 'Crop'
    attrs = {'name': operator.full_name}
    border = operator.raw_operator.crop.cropAmounts.borderAmounts
    left_border, top_border, right_border, bottom_border = (None,) * 4
    if len(border):
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

    attrs['border'] = [left_border, top_border, right_border, bottom_border]

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('crop', convert_crop)

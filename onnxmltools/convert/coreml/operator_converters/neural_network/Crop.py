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
    left, top, right, bottom = (None,) * 4
    if len(border):
        left = border[1].startEdgeSize
        top = border[0].startEdgeSize
        bottom = border[0].endEdgeSize
    else:
        offset = operator.raw_operator.crop.offset
        shape = operator.outputs[0].type.shape
        left = offset[0]
        top = offset[1]
        right = offset[0] + shape[2]
        bottom = offset[1] + shape[3]

    attrs['border'] = [left, top, right, bottom]

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('crop', convert_crop)

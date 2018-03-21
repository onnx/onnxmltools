# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...registration import register_converter


def convert_crop(scope, operator, container):
    if len(operator.input_full_names) > 2:
        raise RuntimeError('Unlike CoreML, ONNX only supports cropping with a single input')

    op_type = 'Crop'
    attrs = {'name': operator.full_name}
    border = operator.raw_operator.crop.cropAmounts.borderAmounts
    left = border[1].startEdgeSize
    top = border[0].startEdgeSize
    right = border[1].endEdgeSize
    bottom = border[0].endEdgeSize

    attrs['border'] = [left, top, right, bottom]

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('crop', convert_crop)

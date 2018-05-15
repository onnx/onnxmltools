# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ....common._registration import register_converter


def convert_crop(scope, operator, container):
    if len(operator.input_full_names) > 2:
        raise RuntimeError('Unlike CoreML, ONNX only supports cropping with a single input')

    op_type = 'Slice'
    attrs = {'name': operator.full_name}
    attrs['axes'] = [2, 3]  # Only do cropping along H- (indexed by 2) and W-axes (indexed by 3)

    if len(operator.inputs) == 1:
        # Mode 1: only one input is provided
        H = operator.inputs[0].type.shape[2]
        W = operator.inputs[0].type.shape[3]
        border = operator.raw_operator.crop.cropAmounts.borderAmounts
        left = border[1].startEdgeSize
        top = border[0].startEdgeSize
        right = border[1].endEdgeSize
        bottom = border[0].endEdgeSize
        attrs['starts'] = [top, left]
        attrs['ends'] = [H - bottom, W - right]
    else:
        # Mode 2: we got two inputs. The second one provides the reference shape (along H- and W-axes).
        H_ref = operator.inputs[1].type.shape[2]
        W_ref = operator.inputs[1].type.shape[3]
        offset_h = operator.raw_operator.crop.offset[0]
        offset_w = operator.raw_operator.crop.offset[1]
        attrs['starts'] = [offset_h, offset_w]
        attrs['ends'] = [offset_h + H_ref, offset_w + W_ref]

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('crop', convert_crop)

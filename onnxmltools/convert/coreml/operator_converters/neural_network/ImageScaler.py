# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_converter


def convert_preprocessing_scaler(scope, operator, container):
    params = operator.raw_operator.scaler
    # Specify some of this operator's attribute. The scale parameter in CoreML is always a scalar.
    # We just copy it and let ONNX scaler to broadcast it to all channels.

    attrs = {'name': operator.full_name, 'scale': params.channelScale}
    color_space = operator.inputs[0].type.color_space
    if color_space == 'GRAY':
        attrs['bias'] = [params.grayBias]
    elif color_space == 'RGB':
        attrs['bias'] = [params.redBias, params.greenBias, params.blueBias]
    elif color_space == 'BGR':
        attrs['bias'] = [params.blueBias, params.greenBias, params.redBias]
    else:
        raise ValueError('Unknown color space for tensor {}'.format(operator.inputs[0].full_name))

    container.add_node('ImageScaler', [operator.inputs[0].full_name], [operator.outputs[0].full_name], **attrs)


register_converter('scalerPreprocessor', convert_preprocessing_scaler)

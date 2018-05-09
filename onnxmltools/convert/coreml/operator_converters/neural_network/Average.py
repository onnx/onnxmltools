# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


from distutils.version import StrictVersion
from ....common._registration import register_converter


def convert_average(scope, operator, container):
    attrs = {'name': operator.full_name}
    if container.targeted_onnx_version < StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0] * len(operator.inputs)
        op_version = 1
    else:
        op_version = 6

    container.add_node('Mean', operator.input_full_names, operator.output_full_names, op_version=op_version, **attrs)


register_converter('average', convert_average)

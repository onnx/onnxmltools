# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# The converters which can be invoked in the conversion framework defined in _topology.py
_converter_pool = {}
# The shape calculators which can be invoked in the conversion framework defined in _topology.py
_shape_calculator_pool = {}


def register_converter(operator_name, conversion_function, overwrite=False):
    if not overwrite and operator_name in _converter_pool:
        raise ValueError('We do not overwrite registrated converter by default')
    _converter_pool[operator_name] = conversion_function


def get_converter(operator_name):
    if operator_name not in _converter_pool:
        raise ValueError('Unsupported conversion for operator %s' % operator_name)
    return _converter_pool[operator_name]


def register_shape_calculator(operator_name, calculator_function, overwrite=False):
    if not overwrite and operator_name in _shape_calculator_pool:
        raise ValueError('We do not overwrite registrated shape calculator by default')
    _shape_calculator_pool[operator_name] = calculator_function


def get_shape_calculator(operator_name):
    if operator_name not in _shape_calculator_pool:
        raise ValueError('Unsupported shape calculation for operator %s' % operator_name)
    return _shape_calculator_pool[operator_name]

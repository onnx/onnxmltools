#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

_converter_map = {}
_nn_converter_map = {}


def register_converter(operation_class, converter_class):
    _converter_map[operation_class] = converter_class


def get_converter(operation_class):
    if operation_class not in _converter_map:
        raise Exception('Operator is not supported: {0}'.format(operation_class))
    return _converter_map[operation_class]


def register_nn_converter(operation_class, converter_class):
    _nn_converter_map[operation_class] = converter_class


def get_nn_converter(operation_class):
    if operation_class not in _nn_converter_map:
        raise Exception('Operator is not supported: {0}'.format(operation_class))
    return _nn_converter_map[operation_class]


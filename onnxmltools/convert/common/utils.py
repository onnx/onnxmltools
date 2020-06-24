# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxconverter_common.utils import *  # noqa


def hummingbird_installed():
    """
    Checks that *Hummingbird* is available.
    """
    try:
        import hummingbird.ml  # noqa: F401

        return True
    except ImportError:
        return False

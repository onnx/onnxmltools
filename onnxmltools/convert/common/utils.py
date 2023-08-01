# SPDX-License-Identifier: Apache-2.0

try:
    from onnxconverter_common.utils import hummingbird_installed  # noqa
except ImportError:
    def hummingbird_installed():
        """
        Checks that *Hummingbird* is available.
        """
        try:
            import hummingbird.ml  # noqa: F401

            return True
        except ImportError:
            return False


def tf2onnx_installed():
    """
    Checks that *tf2onnx* is available.
    """
    try:
        import tf2onnx  # noqa F401
        return True
    except ImportError:
        return False


from onnxconverter_common.utils import *  # noqa

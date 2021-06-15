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

from onnxconverter_common.utils import *  # noqa

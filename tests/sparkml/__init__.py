# SPDX-License-Identifier: Apache-2.0

try:
    pass
except ImportError as e:
    import os

    raise ImportError(
        "Unable to import local test submodule "
        "'tests.sparkml.sparkml_test_base'. "
        "Current directory: %r, PYTHONPATH=%r, in folder=%r."
        % (os.getcwd(), os.environ.get("PYTHONPATH", "-"), os.listdir("."))
    ) from e

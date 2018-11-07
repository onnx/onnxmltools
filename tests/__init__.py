#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------
"""
Releases onnxmltools test as a separate package.

To install it:

::

    pip install onnxmltools-tests
    
To run the tests:

::

    from onnxmltools.tests import run_tests
    run_tests()
"""

__version__ = "0.1"
__author__ = "Microsoft"

from .main import run_tests

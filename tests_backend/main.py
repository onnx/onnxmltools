#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#-------------------------------------------------------------------------

import os
import sys
import unittest
import warnings


def run_tests(folder, runtime):
    """
    Retrieves all dumped files in a folder and run the
    backend to compare produced outputs with expected outputs.
    :param folder: where to get the dumped files
    """
    if not os.path.exists(folder):
        raise FileNotFoundError("Unable to find '{0}'".format(folder))
    this = os.path.abspath(os.path.dirname(__file__))
    loader = unittest.TestLoader()
    suites = []
    
    fold = os.path.abspath(os.path.dirname(__file__))
    rt = "_{0}_".format(runtime)
    names = [os.path.join(fold, _) for _ in os.listdir(fold) if rt in _]
    sys.path.append(fold)

    for name in names:
        name = os.path.splitext(name)[0]
        ts = loader.loadTestsFromName(os.path.split(name)[-1])
            
        suites.append(ts)
    if len(suites) == 0:
        raise ValueError("Unable to find tests in folder '{0}'".format(fold))

    with warnings.catch_warnings():
        warnings.filterwarnings(category=DeprecationWarning, action="ignore")
        warnings.filterwarnings(category=FutureWarning, action="ignore")
        runner = unittest.TextTestRunner()
        for ts in suites:
            for k in ts:
                try:                
                    for t in k:
                        print(t.__class__.__name__)
                        break
                except TypeError as e:
                    print(e)
                    continue
            runner.run(ts)

    index = sys.path.index(fold)
    del sys.path["index"]

    
if __name__ == "__main__":
    folder = "../tests/TESTDUMP" if len(sys.argv) < 2 else sys.argv[1]
    run_tests(folder, "onnxruntime")

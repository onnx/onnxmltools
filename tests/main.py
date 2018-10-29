#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#-------------------------------------------------------------------------

import os
import sys
import unittest
import warnings


def run_tests(library=None, folder=None):
    """
    Runs all unit tests or unit tests specific to one library.
    The tests produce a series of files dumped into ``folder``
    which can be later used to tests a backend (or a runtime).
    
    :param library: possible options,
        ``'Sklearn'`` for *scikit-learn*,
        ``'LightGbm'`` for *lightgbm*,
        ``'Cml'`` for *coremltools*,
        ``'Keras'`` for *keras*,
        parameter *library* can be None to test all,
        a list of them or just a string
    :param folder: where to put the dumped files
    """
    if folder is None:
        folder = 'TESTDUMP'
    os.environ["ONNXTESTDUMP"] = folder
    
    
    try:
        import onnxmltools
    except ImportError:
        raise ImportError("Cannot import onnxmltools. It must be installed first.")
    
    available = {'Sklearn': ['sklearn'],
                 'LightGbm': ['lightgbm'],
                 'Cml': ['coreml'],
                 'Keras': ['end2end']}
    
    if library is None:
        library = list(available.keys())
    elif not isinstance(library, list):
        library = [library]
    for lib in library:
        if lib not in available:
            raise KeyError("library '{0}' must be in {1}".format(lib, ", ".join(sorted(available))))

    this = os.path.abspath(os.path.dirname(__file__))
    loader = unittest.TestLoader()
    suites = []
    
    for lib in sorted(library):
        subs = available[lib]
        for sub in subs:
            fold = os.path.join(this, sub)
            if not os.path.exists(fold):
                raise FileNotFoundError("Unable to find '{0}'".format(fold))
            
            # ts = loader.discover(fold)
            sys.path.append(fold)
            names = [_ for _ in os.listdir(fold) if _.startswith("test")]
            for name in names:
                name = os.path.splitext(name)[0]
                ts = loader.loadTestsFromName(name)
                suites.append(ts)
            index = sys.path.index(fold)
            del sys.path[index]
    
    with warnings.catch_warnings():
        warnings.filterwarnings(category=DeprecationWarning, action="ignore")
        warnings.filterwarnings(category=FutureWarning, action="ignore")
        runner = unittest.TextTestRunner()
        for ts in suites:
            for k in ts:
                for t in k:
                    print(t.__class__.__name__)
                    break
            runner.run(ts)
    
    from onnxmltools.utils.tests_helper import make_report_backend
    report = make_report_backend(folder)
    
    from pandas import DataFrame, set_option
    set_option("display.max_columns", None)
    set_option("display.max_rows", None)
    
    df = DataFrame(report).sort_values(["_model"])
    
    import onnx
    import onnxruntime
    print(df)
    df["onnx-version"] = onnx.__version__
    df["onnxruntime-version"] = onnxruntime.__version__
    df.to_excel(os.path.join(folder, "report_backend.xlsx"))
                    
    
if __name__ == "__main__":
    folder = None if len(sys.argv) < 2 else sys.argv[1]
    run_tests(folder=folder)

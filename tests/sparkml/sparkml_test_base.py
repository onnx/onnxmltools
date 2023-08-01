# SPDX-License-Identifier: Apache-2.0

'''
Testcase Base class for SparkML tests
'''
import os
import inspect
import unittest
from tests.sparkml.sparkml_test_utils import start_spark, stop_spark


class SparkMlTestCase(unittest.TestCase):
    def _get_spark_options(self):
        return None

    def setUp(self):
        if os.name == 'nt' and os.environ.get('HADOOP_HOME') is None:
            this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            print('setting HADOOP_HOME to: ', this_script_dir)
            os.environ['HADOOP_HOME'] = this_script_dir
        self.spark = start_spark(self._get_spark_options())

    def tearDown(self):
        stop_spark(self.spark)


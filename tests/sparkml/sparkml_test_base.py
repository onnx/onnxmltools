'''
Testcase Base class for SparkML tests
'''
import unittest
from tests.sparkml.sparkml_test_utils import start_spark, stop_spark


class SparkMlTestCase(unittest.TestCase):
    def setUp(self):
        import os
        import inspect
        if os.name == 'nt' and os.environ.get('HADOOP_HOME') is None:
            this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            print('setting HADOOP_HOME to: ', this_script_dir)
            os.environ['HADOOP_HOME'] = this_script_dir
        self.spark = start_spark()

    def tearDown(self):
        stop_spark(self.spark)


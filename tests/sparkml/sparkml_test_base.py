'''
Testcase Base class for SparkML tests
'''
import unittest
from onnxmltools.utils.sparkml_test_utils import start_spark, stop_spark


class SparkMlTestCase(unittest.TestCase):
    def setUp(self):
        self.spark = start_spark()

    def tearDown(self):
        stop_spark(self.spark)


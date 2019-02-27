"""
Tests SparkML TreeEnsembleClassifier converter.
"""
import unittest
import numpy
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from  pyspark.ml.classification import DecisionTreeClassifier

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_sparkml_model
from sparkml import SparkMlTestCase
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer


class TestSparkmlLinearRegression(SparkMlTestCase):
    def test_model_linear_regression_basic(self):
        df = self.spark.createDataFrame([
            (1.0, Vectors.dense(1.0)),
            (0.0, Vectors.sparse(1, [], []))],
            ["label", "features"]
        )
        stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
        si_model = stringIndexer.fit(df)
        td = si_model.transform(df)
        dt = DecisionTreeClassifier(maxDepth=2, labelCol="indexed")
        model = dt.fit(td)
        print(model.numClasses)


if __name__ == "__main__":
    unittest.main()

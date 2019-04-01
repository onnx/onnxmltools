import pandas
import sys
import unittest

import numpy
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlPCA(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_polynomial_expansion(self):
        data = self.spark.createDataFrame([
            (Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
            (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
            (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)
        ], ["features"])
        pca = PCA(k=2, inputCol="features", outputCol="pca_features")
        model = pca.fit(data)

        # the input name should match that of what StringIndexer.inputCol
        feature_count = data.first()[0].size
        N = data.count()
        model_onnx = convert_sparkml(model, 'Sparkml PCA', [('features', FloatTensorType([N, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().pca_features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlPCA")


if __name__ == "__main__":
    unittest.main()

import sys
import unittest

import pandas
import numpy
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmMinHashLSH(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_min_hash_lsh(self):
        data = self.spark.createDataFrame([
            (0, Vectors.sparse(6, [0, 1, 2], [1.0, 1.0, 1.0]),),
            (1, Vectors.sparse(6, [2, 3, 4], [1.0, 1.0, 1.0]),),
            (2, Vectors.sparse(6, [0, 2, 4], [1.0, 1.0, 1.0]),)
        ], ["id", "features"])
        mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
        model = mh.fit(data)

        feature_count = data.first()[1].size
        model_onnx = convert_sparkml(model, 'Sparkml MinHashLSH', [
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data.limit(1))
        predicted.show(100, False)
        data_np = data.limit(1).toPandas().features.apply(
            lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [
            predicted.toPandas().hashes.apply(lambda x: pandas.Series(x)
                                              .map(lambda y: y.values[0])).values.astype(numpy.float32),
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlMinHashLSH")


if __name__ == "__main__":
    unittest.main()

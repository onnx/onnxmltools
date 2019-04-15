import sys
import unittest

import pandas
import numpy
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestBucketedRandomProjectionLSH(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_bucketed_random_projection_lsh(self):
        data = self.spark.createDataFrame([
            (0, Vectors.dense([-1.0, -1.0 ]),),
            (1, Vectors.dense([-1.0, 1.0 ]),),
            (2, Vectors.dense([1.0, -1.0 ]),),
            (3, Vectors.dense([1.0, 1.0]),)
        ], ["id", "features"])
        mh = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", seed=12345, bucketLength=1.0)
        model = mh.fit(data)

        feature_count = data.first()[1].size
        model_onnx = convert_sparkml(model, 'Sparkml BucketedRandomProjectionLSH', [
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(
            lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [
            predicted.toPandas().hashes.apply(lambda x: pandas.Series(x)
                                              .map(lambda y: y.values[0])).values.astype(numpy.float32),
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlBucketedRandomProjectionLSH")


if __name__ == "__main__":
    unittest.main()

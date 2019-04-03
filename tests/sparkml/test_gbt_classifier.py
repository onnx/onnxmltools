import sys
import inspect
import unittest
from distutils.version import StrictVersion

import onnx
import pandas
import numpy
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.linalg import VectorUDT, SparseVector, Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType, FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model
from pyspark.ml.feature import StringIndexer, VectorIndexer


class TestSparkmTreeEnsembleClassifier(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.3'), 'Need Greater Opset 9')
    def test_gbt_classifier(self):
        raw_data = self.spark.createDataFrame([
            (1.0, Vectors.dense(1.0)),
            (0.0, Vectors.sparse(1, [], []))
        ], ["label", "features"])
        string_indexer = StringIndexer(inputCol="label", outputCol="indexed")
        si_model = string_indexer.fit(raw_data)
        data = si_model.transform(raw_data)
        gbt = GBTClassifier(maxIter=5, maxDepth=2, labelCol="indexed", seed=42)
        model = gbt.fit(data)
        feature_count = data.first()[1].size
        model_onnx = convert_sparkml(model, 'Sparkml GBT Classifier', [
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlGBTClassifier")


if __name__ == "__main__":
    unittest.main()

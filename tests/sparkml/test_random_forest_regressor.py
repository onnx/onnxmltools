import sys
import inspect
import unittest
from distutils.version import StrictVersion

import onnx
import pandas
import numpy
from pyspark.ml.linalg import VectorUDT, SparseVector
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType, StringTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model
from pyspark.ml.feature import VectorIndexer, StringIndexer


class TestSparkmTreeEnsembleRegressor(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.3'), 'Need Greater Opset 9')
    def test_random_forrest_regression(self):
        import os
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "sample_libsvm_data.txt")
        original_data = self.spark.read.format("libsvm").load(input_path)
        #
        # truncate the features
        #
        feature_count = 5
        self.spark.udf.register("truncateFeatures",
                                lambda x: SparseVector(feature_count, range(0,feature_count), x.toArray()[125:130]),
                                VectorUDT())
        data = original_data.selectExpr("cast(label as string) as label", "truncateFeatures(features) as features")
        label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
        feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",
                                        maxCategories=10, handleInvalid='keep')

        rf = RandomForestRegressor(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
        pipeline = Pipeline(stages=[label_indexer, feature_indexer, rf])
        model = pipeline.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml RandomForest Regressor', [
            ('label', StringTensorType([1, 1])),
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        data_np = {
            'label': data.toPandas().label.values,
            'features': data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        }
        expected = [
            predicted.toPandas().indexedLabel.values.astype(numpy.int64),
            predicted.toPandas().prediction.values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlRandomForestRegressor")


if __name__ == "__main__":
    unittest.main()

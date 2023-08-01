# SPDX-License-Identifier: Apache-2.0

import sys
import inspect
import unittest
from distutils.version import StrictVersion
import os
import onnx
import pandas
import numpy
from pyspark.ml.linalg import VectorUDT, SparseVector
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType, StringTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase
from pyspark.ml.feature import VectorIndexer, StringIndexer


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmRandomForestRegressor(SparkMlTestCase):

    @unittest.skipIf(sys.platform == 'win32',
                     reason="UnsatisfiedLinkError")
    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.3'),
                     'Need Greater Opset 9')
    def test_random_forest_regression(self):
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
                                        maxCategories=10, handleInvalid='error')

        rf = RandomForestRegressor(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
        pipeline = Pipeline(stages=[label_indexer, feature_indexer, rf])
        model = pipeline.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml RandomForest Regressor', [
            ('label', StringTensorType([None, 1])),
            ('features', FloatTensorType([None, feature_count]))
        ], spark_session=self.spark, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data.limit(1))
        data_np = {
            'label': data.limit(1).toPandas().label.values.reshape((-1, 1)),
            'features': data.limit(1).toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        }
        expected = [
            predicted.toPandas().indexedLabel.values.astype(numpy.int64),
            predicted.toPandas().prediction.values.astype(numpy.float32)
        ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlRandomForestRegressor")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['indexedLabel', 'prediction'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()

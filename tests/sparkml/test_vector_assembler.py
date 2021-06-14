# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.feature import VectorAssembler
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlVectorAssembler(SparkMlTestCase):

    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_model_vector_assembler(self):
        col_names = ["a", "b", "c"]
        model = VectorAssembler(inputCols=col_names, outputCol='features')
        data = self.spark.createDataFrame([(1., 0., 3.)], col_names)
        model_onnx = convert_sparkml(model, 'Sparkml VectorAssembler',  [
            ('a', FloatTensorType([None, 1])),
            ('b', FloatTensorType([None, 1])),
            ('c', FloatTensorType([None, 1]))
        ])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(data)
        expected = predicted.select("features").toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values
        data_np = {
            'a': data.select('a').toPandas().values.astype(numpy.float32),
            'b': data.select('b').toPandas().values.astype(numpy.float32),
            'c': data.select('c').toPandas().values.astype(numpy.float32)
        }
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlVectorAssembler")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['features'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()

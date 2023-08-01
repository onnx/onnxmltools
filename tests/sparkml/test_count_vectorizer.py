# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml.sparkml_test_utils import (
    save_data_models,
    run_onnx_model,
    compare_results,
)
from tests.sparkml import SparkMlTestCase

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmlCountVectorizer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_count_vectorizer_default(self):
        data = self.spark.createDataFrame(
            [
                ("A B C".split(" "),),
                ("A B B C A".split(" "),),
            ],
            ["text"],
        )
        count_vec = CountVectorizer(
            inputCol="text", outputCol="result", minTF=1.0, binary=False
        )
        model: CountVectorizerModel = count_vec.fit(data)
        result = model.transform(data)

        model_onnx = convert_sparkml(
            model,
            "Sparkml CountVectorizer",
            [("text", StringTensorType([None, None]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        data_pd = data.toPandas()
        data_np = {
            "text": data_pd.text.apply(lambda x: pandas.Series(x)).values.astype(str),
        }

        expected = {
            "prediction_result": numpy.asarray(
                result.toPandas()
                .result.apply(lambda x: pandas.Series(x.toArray()))
                .values.astype(numpy.float32)
            ),
        }

        paths = save_data_models(
            data_np,
            expected,
            model,
            model_onnx,
            basename="SparkmlCountVectorizerModel_Default",
        )
        onnx_model_path = paths[-1]

        output_names = ["result"]
        output, output_shapes = run_onnx_model(output_names, data_np, onnx_model_path)
        actual_output = dict(zip(output_names, output))

        assert output_shapes[0] == [None, 3]
        compare_results(
            expected["prediction_result"], actual_output["result"], decimal=5
        )

    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_count_vectorizer_binary(self):
        data = self.spark.createDataFrame(
            [
                ("A B C".split(" "),),
                ("A B B C A".split(" "),),
                ("B B B D".split(" "),),
            ],
            ["text"],
        )
        count_vec = CountVectorizer(
            inputCol="text", outputCol="result", minTF=2.0, binary=True
        )
        model: CountVectorizerModel = count_vec.fit(data)
        result = model.transform(data)

        model_onnx = convert_sparkml(
            model,
            "Sparkml CountVectorizer",
            [("text", StringTensorType([None, None]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        data_pd = data.toPandas()
        data_np = {
            "text": data_pd.text.apply(lambda x: pandas.Series(x)).values.astype(str),
        }

        expected = {
            "prediction_result": numpy.asarray(
                result.toPandas()
                .result.apply(lambda x: pandas.Series(x.toArray()))
                .values.astype(numpy.float32)
            ),
        }

        paths = save_data_models(
            data_np,
            expected,
            model,
            model_onnx,
            basename="SparkmlCountVectorizerModel_Binary",
        )
        onnx_model_path = paths[-1]

        output_names = ["result"]
        output, output_shapes = run_onnx_model(output_names, data_np, onnx_model_path)
        actual_output = dict(zip(output_names, output))

        assert output_shapes[0] == [None, 4]
        compare_results(
            expected["prediction_result"], actual_output["result"], decimal=5
        )


if __name__ == "__main__":
    unittest.main()

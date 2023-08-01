# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.feature import Word2Vec
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


## For some reason during the spark bring up and shutdown something happens causing these
## tests to fail. For that you need to run each test here individually
## For now these will be commented out so as not to break the build
##      AttributeError: 'NoneType' object has no attribute 'setCallSite' on model.surrogateDF
##  Therefore we leave these tests out for now until a newere version of pyspark is availabe that address this issue
class TestSparkmlWord2Vec(SparkMlTestCase):

    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_word2vec(self):
        data = self.spark.createDataFrame([
            ("Hi I heard about Spark".split(" "), ),
            ("I wish Java could use case classes".split(" "), ),
            ("Logistic regression models are neat".split(" "), )
        ], ["text"])
        word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
        model = word2Vec.fit(data)
        vectors = model.getVectors()
        vectors.show(100, False)
    
        result = model.transform(data)
        result.show(100, False)
    
        # the input name should match that of inputCol
        feature_count = len(data.first()[0])
        model_onnx = convert_sparkml(model, 'Sparkml Word2Vec', [('text', StringTensorType([None, feature_count]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data.limit(1))
        expected = predicted.toPandas().result.apply(lambda  x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.limit(1).toPandas().text.values
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlWord2Vec")
        onnx_model_path = paths[-1]
        data_np = numpy.array(data_np[0]).reshape((1, -1))
        output, output_shapes = run_onnx_model(['result'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()

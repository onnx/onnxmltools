"""
Tests scikit-onehotencoder converter.
"""
import unittest
import numpy
from sklearn.preprocessing import OneHotEncoder
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from onnxmltools.utils import dump_data_and_model


class TestSklearnOneHotEncoderConverter(unittest.TestCase):

    def test_model_one_hot_encoder(self):
        # categorical_features will be removed in 0.22 (this test will fail by then).
        # FutureWarning: The handling of integer data will change in version 0.22.
        # Currently, the categories are determined based on the range [0, max(values)],
        # while in the future they will be determined based on the unique values.
        # If you want the future behaviour and silence this warning,
        # you can specify "categories='auto'".
        model = OneHotEncoder()
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=numpy.int64)
        model.fit(data)
        model_onnx = convert_sklearn(model, 'scikit-learn one-hot encoder',
                                     [('input', Int64TensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnOneHotEncoderInt64-SkipDim1")

    def test_one_hot_encoder_mixed_string_int(self):
        # categorical_features will be removed in 0.22 (this test will fail by then).
        data = [["0.4", "0.2", 3], ["1.4", "1.2", 0], ["0.2", "2.2", 1]]
        model = OneHotEncoder(categories='auto')        
        model.fit(data)
        inputs = [('input1', StringTensorType([1, 2])), ('input2', Int64TensorType([1, 1]))]
        model_onnx = convert_sklearn(model, 'one-hot encoder mixed-type inputs', inputs)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnOneHotEncoderStringInt64",
                            allow_failure=True)


if __name__ == "__main__":
    unittest.main()

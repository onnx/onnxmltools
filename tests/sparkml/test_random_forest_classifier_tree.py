# SPDX-License-Identifier: Apache-2.0

import sys
import inspect
import unittest
import os
import packaging.version as pv
import onnx
import numpy
from numpy.random import randint
from onnxruntime import InferenceSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmRandomForestClassifierTree(SparkMlTestCase):
    @unittest.skipIf(sys.platform == "win32", reason="UnsatisfiedLinkError")
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    @unittest.skipIf(
        pv.Version(onnx.__version__) <= pv.Version("1.3"), "Need Greater Opset 9"
    )
    def test_random_forest_classification_tree(self):
        FEATURE_LEN = 32

        def infer_from_onnx(model_onnx, input_list):
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            input_name = sess.get_inputs()[0].name
            pred_onx = sess.run(
                None, {input_name: numpy.array(input_list, numpy.float32)}
            )
            return pred_onx

        def export_as_onnx(model):
            model_onnx = convert_sparkml(
                model,
                "Phish Classifier",
                [("features", FloatTensorType([None, FEATURE_LEN]))],
                spark_session=self.spark,
                target_opset=TARGET_OPSET,
            )
            return model_onnx

        def create_model(input_path):
            df = self.spark.read.csv(input_path, header=True, inferSchema=True)

            vec_assembler = VectorAssembler(
                inputCols=["c" + str(i) for i in range(FEATURE_LEN)],
                outputCol="features",
            )

            data = vec_assembler.transform(df)
            rf = RandomForestClassifier(
                labelCol="label", featuresCol="features", numTrees=5
            )
            model = rf.fit(dataset=data)  # RandomForestClassificationModel
            # model.save("./dummy_spark_model/model/")
            return model

        this_script_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        input_path = os.path.join(this_script_dir, "data", "features_32.csv")
        model = create_model(input_path)
        model_onnx = export_as_onnx(model)

        input_list = [[randint(0, 20) for _ in range(32)]]
        pred_onx = infer_from_onnx(model_onnx, input_list)
        self.assertEqual(len(pred_onx), 2)
        # print(pred_onx)


if __name__ == "__main__":
    unittest.main()

..  SPDX-License-Identifier: Apache-2.0


onnxmltools: Convert your model into ONNX
=========================================

ONNXMLTools enables you to convert models from different machine learning
toolkits into `ONNX <https://onnx.ai>`_.
Currently the following toolkits are supported:

* `Apple Core ML <https://developer.apple.com/documentation/coreml>`_,
  (`onnx-coreml <https://github.com/onnx/onnx-coreml>`_ does the reverse
  conversion from *onnx* to *Apple Core ML*) (up to version 3.1)
* `catboost <https://catboost.ai/>`_
* `h2o <http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html>`_
  (a subset only)
* `Keras <https://keras.io/>`_
* `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`_
* `libsvm <https://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_
* `scikit-learn <https://scikit-learn.org/stable/>`_
  (subset of models convertible to ONNX)
* `SparkML <https://spark.apache.org/docs/latest/ml-guide.html>`_
* `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_

*onnxmltools* leverages existing converting library,
`sklearn-onnx <https://github.com/onnx/sklearn-onnx>`_,
`tensorflow-onnx <https://github.com/onnx/tensorflow-onnx>`_
and implements converters for the other libraries.

.. toctree::
    :maxdepth: 2

    tutorial
    api_summary
    auto_examples/index

*onnxmltools* converts models in ONNX format which
can be then used to compute predictions with the
backend of your choice. Every converter is tested with:
`onnxruntime <https://pypi.org/project/onnxruntime/>`_
(does also exist with GPU:
`onnxruntime-gpu <https://pypi.org/project/onnxruntime-gpu>`_).
Here is a typical example which trains a model, converts into
ONNX and finally uses *onnxruntime* to predict.

::

    # Train a model.
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert into ONNX format with onnxmltools
    from onnxmltools import convert_sklearn
    from onnxmltools.utils import save_model
    from onnxconverter_common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    save_model(onx, "rf_iris.onnx")

    # Compute the prediction with ONNX Runtime
    import onnxruntime as rt
    import numpy
    sess = rt.InferenceSession("rf_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

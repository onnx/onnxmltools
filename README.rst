
.. image:: https://github.com/onnx/onnxmltools/blob/master/docs/ONNXMLTools_logo_main.png

Introduction 
============

ONNXMLTools enables you to convert models from different machine 
learning toolkits into `ONNX <https://onnx.ai>`_. 
Currently the following toolkits are supported:

* Apple CoreML
* scikit-learn
  (subset of models convertible to ONNX)

Install
=======

::

    pip install onnxmltools

Dependancies
============

`scikit-learn <http://scikit-learn.org/stable/>`_ is needed to convert
a scikit-learn model, `coremltools <https://pypi.python.org/pypi/coremltools>`_
for Apple CoreML.

Example
=======

Here is a simple example to convert a CoreML model:

::

    import onnxmltools
    import coremltools

    model_coreml = coremltools.utils.load_spec("image_recognition.mlmodel")
    model_onnx = onnxmltools.convert.convert_coreml(model_coreml, "Image_Reco")

    # Save as text
    onnxmltools.utils.save_text(model_onnx, "image_recognition.json")

    # Save as protobuf
    onnxmltools.utils.save_model(model_onnx, "image_recognition.onnx")

License
=======

MIT License

Acknowledgments
===============

The initial version of this package was developed by the following 
developers and data scientists at Microsoft during winter 2017: 
Zeeshan Ahmed, Wei-Sheng Chin, Aidan Crook, Xavier Dupre, Costin Eseanu, 
Tom Finley, Lixin Gong, Scott Inglis, Pei Jiang, Ivan Matantsev, 
Prabhat Roy, M. Zeeshan Siddiqui, Shouheng Yi, Shauheen Zahirazami, Yiwen Zhu.

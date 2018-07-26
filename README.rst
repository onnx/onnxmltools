
.. image:: https://github.com/onnx/onnxmltools/raw/master/docs/ONNXMLTools_logo_main.png

.. list-table::
   :widths: 4 4
   :header-rows: 0
   
   * - .. image:: https://travis-ci.org/onnx/onnxmltools.svg?branch=master
            :target: https://travis-ci.org/onnx/onnxmltools
            :alt: Build Status Linux
     - .. image:: https://ci.appveyor.com/api/projects/status/d1xav3amubypje4n?svg=true
            :target: https://ci.appveyor.com/project/xadupre/onnxmltools
            :alt: Build Status Windows

Introduction 
============

ONNXMLTools enables you to convert models from different machine 
learning toolkits into `ONNX <https://onnx.ai>`_. 
Currently the following toolkits are supported:

* Apple Core ML
* scikit-learn (subset of models convertible to ONNX)
* Keras (version 2.0.0 or higher)
* LightGBM (through its scikit-learn interface)

Install
=======

::

    pip install onnxmltools

Dependencies
============

This package uses ONNX, NumPy, and ProtoBuf. If you are converting a model from scikit-learn, Apple Core ML, or Keras you need the following packages installed respectively:
1. `scikit-learn <http://scikit-learn.org/stable/>`_
2. `coremltools <https://pypi.python.org/pypi/coremltools>`_
3. `Keras <https://github.com/keras-team/keras>`_

Example
=======

Here is a simple example to convert a Core ML model:

::

    import onnxmltools
    import coremltools

    model_coreml = coremltools.utils.load_spec('image_recognition.mlmodel')
    model_onnx = onnxmltools.convert_coreml(model_coreml, 'Image_Reco')

    # Save as text
    onnxmltools.utils.save_text(model_onnx, 'image_recognition.json')

    # Save as protobuf
    onnxmltools.utils.save_model(model_onnx, 'image_recognition.onnx')

Next, we show a simple usage of the Keras converter.

::
    import onnxmltools
    from keras.layers import Input, Dense, Add
    from keras.models import Model

    # N: batch size, C: sub-model input dimension, D: final model's input dimension
    N, C, D = 2, 3, 3

    # Define a sub-model, it will become a part of our final model
    sub_input1 = Input(shape=(C,))
    sub_mapped1 = Dense(D)(sub_input1)
    sub_model1 = Model(inputs=sub_input1, outputs=sub_mapped1)

    # Define another sub-model, it will become a part of our final model
    sub_input2 = Input(shape=(C,))
    sub_mapped2 = Dense(D)(sub_input2)
    sub_model2 = Model(inputs=sub_input2, outputs=sub_mapped2)

    # Define our final model
    input1 = Input(shape=(D,))
    input2 = Input(shape=(D,))
    mapped1_2 = sub_model1(input1)
    mapped2_2 = sub_model2(input2)
    sub_sum = Add()([mapped1_2, mapped2_2])
    keras_model = Model(inputs=[input1, input2], output=sub_sum)

    # Convert it!
    onnx_model = onnxmltools.convert_keras(keras_model)

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

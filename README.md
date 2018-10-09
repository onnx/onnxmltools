
<p align="center"><img width="40%" src="docs/ONNXMLTools_logo_main.png" /></p>

| Linux | Windows |
|-------|---------|
| [![Build Status](https://travis-ci.org/onnx/onnxmltools.svg?branch=master)](https://travis-ci.org/onnx/onnxmltools) | [![Build status](https://ci.appveyor.com/api/projects/status/d1xav3amubypje4n?svg=true)](https://ci.appveyor.com/project/xadupre/onnxmltools) |


# Introduction 
ONNXMLTools enables you to convert models from different machine learning toolkits into [ONNX](https://onnx.ai). Currently the following toolkits are supported:
* Apple Core ML
* scikit-learn (subset of models convertible to ONNX)
* Keras (version 2.0.0 or higher)
* LightGBM (through its scikit-learn interface)

(To convert Tensorflow models to ONNX, see [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx))
(To convert ONNX model to Core ML, see [onnx-coreml](https://github.com/onnx/onnx-coreml))

# Getting Started
Clone this repository on your local machine.

## Install
You can install latest release of ONNXMLTools from pypi:
```
pip install onnxmltools
```
or install from source:
```
pip install git+https://github.com/onnx/onnxmltools
```
If you choose to install `onnxmltools` from its source code, you must set an environment variable `ONNX_ML=1` before installing `onnx` package.

## Dependencies
This package uses ONNX, NumPy, and ProtoBuf. If you are converting a model from scikit-learn, Apple Core ML, Keras, or LightGBM, you need the following packages installed respectively:
1. scikit-learn
2. CoreMLTools
3. Keras
4. LightGBM (scikit-learn interface)

## Examples
Here is a simple example to convert a Core ML model:
```python
import onnxmltools
import coremltools

# Load a Core ML model
coreml_model = coremltools.utils.load_spec('example.mlmodel')

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model, 'Example Model')

# Save as text
onnxmltools.utils.save_text(onnx_model, 'example.json')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'example.onnx')
```
Next, we show a simple usage of the Keras converter.
```python
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

# Define a model built upon the previous two sub-models
input1 = Input(shape=(D,))
input2 = Input(shape=(D,))
mapped1_2 = sub_model1(input1)
mapped2_2 = sub_model2(input2)
sub_sum = Add()([mapped1_2, mapped2_2])
keras_model = Model(inputs=[input1, input2], output=sub_sum)

# Convert it!
onnx_model = onnxmltools.convert_keras(keras_model)
```


# License
[MIT License](LICENSE)

## Acknowledgments
The package was developed by the following engineers and data scientists at Microsoft starting from winter 2017: Zeeshan Ahmed, Wei-Sheng Chin, Aidan Crook, Xavier Dupre, Costin Eseanu, Tom Finley, Lixin Gong, Scott Inglis, Pei Jiang, Ivan Matantsev, Prabhat Roy, M. Zeeshan Siddiqui, Shouheng Yi, Shauheen Zahirazami, Yiwen Zhu, Du Li, Xuan Li, Wenbing Li

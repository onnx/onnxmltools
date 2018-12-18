
<p align="center"><img width="40%" src="docs/ONNXMLTools_logo_main.png" /></p>

| Linux | Windows |
|-------|---------|
| [![Build Status](https://travis-ci.org/onnx/onnxmltools.svg?branch=master)](https://travis-ci.org/onnx/onnxmltools) | [![Build status](https://ci.appveyor.com/api/projects/status/d1xav3amubypje4n?svg=true)](https://ci.appveyor.com/project/xadupre/onnxmltools) |

# Introduction 
ONNXMLTools enables you to convert models from different machine learning toolkits into [ONNX](https://onnx.ai). Currently the following toolkits are supported:
* Apple Core ML
* scikit-learn (subset of models convertible to ONNX)
* Keras
* LightGBM (through its scikit-learn interface)

To convert Tensorflow models to ONNX, see [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx).

## Install
You can install latest release of ONNXMLTools from [PyPi](https://pypi.org/project/onnxmltools/):
```
pip install onnxmltools
```
or install from source:
```
pip install git+https://github.com/onnx/onnxmltools
```
If you choose to install `onnxmltools` from its source code, you must set the environment variable `ONNX_ML=1` before installing the `onnx` package. 

## Dependencies
This package relies on ONNX, NumPy, and ProtoBuf. If you are converting a model from scikit-learn, Core ML, Keras, or LightGBM, you will need an environment with the respective package installed from the list below:
1. scikit-learn
2. CoreMLTools
3. Keras (version 2.0.8 or higher) with the corresponding Tensorflow version
4. LightGBM (scikit-learn interface)

# Examples

## Converting Models 
If you want the converted ONNX model to be compatible with a certain ONNX version, please specify the target_opset parameter upon invoking the convert function. The following Keras model conversion example demonstrates this below. You can identify the mapping from ONNX Operator Sets (referred to as opsets) to ONNX versions in the [versioning documentation](https://github.com/onnx/onnx/blob/master/docs/Versioning.md#released-versions). 

### CoreML -> ONNX Conversion
Here is a simple code snippet to convert a Core ML model into an ONNX model.

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

### Keras -> ONNX Conversion
Next, we show an example of converting a Keras model into an ONNX model with `target_opset=7`, which corresponds to ONNX version 1.2.

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

# Convert it! The target_opset parameter is optional.
onnx_model = onnxmltools.convert_keras(keras_model, target_opset=7) 

```

### Checking the ONNX Operator Set (opset) of your converted model

You can check the opset of your ONNX model using [Netron](https://github.com/lutzroeder/Netron), a viewer for Neural Network models. Alternatively, you could identify your converted model's opset version through the following line of code:

```
# add line here
```

If the result from checking your ONNX model's opset above is smaller than the `target_opset` number you passed into the onnxmltools.convert function, do not be worried. Note that the ONNXMLTools converter works by looking at each operator in your original framework's model and identifying the respective ONNX opset in which it has most recently been updated. It then takes the maximum over all of the operators used in your new ONNX model to result in the converted model's opset number.

Let's take a model with two operators. If Operator A was most recently updated in Opset 6, and Operator B was most recently updated in Opset 7, the ONNX model's opset will always be 7, even if you request target_opset=8. Documentation for the [ONNX Model format] and more examples for converting models from different frameworks can be found in the [ONNX tutorials](https://github.com/onnx/tutorials) repository. 

# Testing model converters

*onnxmltools* converts models into the ONNX format which
can be then used to compute predictions with the
backend of your choice. There exists a way
to automatically check every converter with
[onnxruntime](https://pypi.org/project/onnxruntime/) or
[onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/).

## Test all existing converters

This process requires the user to clone the *onnxmltools* repository.
The following command runs all unit tests and generates
dumps of models, inputs, expected outputs and converted models
in folder ``TESTDUMP``.

```
python tests/main.py DUMP
```

It requires *onnxruntime*, *numpy* for most models,
*pandas* for transforms related to text features, and
*scipy* for sparse features. One test also requires
*keras* to test a custom operator. That means
*sklearn* or any machine learning library is requested.

## Add a new converter

Once the converter is implemented, a unit test is added
to confirm that it works. At the end of the unit test, function
*dump_data_and_model* or any equivalent function must be called
to dump the expected output and the converted model.
Once these file are generated, a corresponding test must
be added in *tests_backend* to compute the prediction
with the runtime.

# License
[MIT License](LICENSE)


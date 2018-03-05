
<p align="center"><img width="40%" src="docs/ONNXMLTools_logo_main.png" /></p>

| Linux | Windows |
|-------|---------|
| [![Build Status](https://travis-ci.org/onnx/onnxmltools.svg?branch=master)](https://travis-ci.org/onnx/onnxmltools) | [![Build status](https://ci.appveyor.com/api/projects/status/d1xav3amubypje4n?svg=true)](https://ci.appveyor.com/project/xadupre/onnxmltools) |


# Introduction 
ONNXMLTools enables you to convert models from different machine learning toolkits into [ONNX](https://onnx.ai). Currently the following toolkits are supported:
* Apple CoreML
* scikit-learn (subset of models convertible to ONNX)

(To convert ONNX model to CoreML, see [onnx-coreml](https://github.com/onnx/onnx-coreml))

# Getting Started
Clone this repository on your local machine.

## Install
Currently you can install ONNXMLTools from source:
```
pip install git+https://github.com/onnx/onnxmltools
```

## Dependancies
This package uses NumPy as well as ProtoBuf. Also If you are converting a model from Scikit-learn or Apple CoreML you need the following packages installed respectively:
1. scikit-learn
2. CoreMLTools

## Example
Here is a simple example to convert a CoreML model:
```
import onnxmltools
import coremltools

model_coreml = coremltools.utils.load_spec("image_recognition.mlmodel")
model_onnx = onnxmltools.convert.convert_coreml(model_coreml, "Image_Reco")

# Save as text
onnxmltools.utils.save_text(model_onnx, "image_recognition.json")

# Save as protobuf
onnxmltools.utils.save_model(model_onnx, "image_recognition.onnx")
```


# License
[MIT License](LICENSE)

## Acknowledgments
The initial version of this package was developed by the following engineers and data scientists at Microsoft during winter 2017: Zeeshan Ahmed, Wei-Sheng Chin, Aidan Crook, Xavier Dupre, Costin Eseanu, Tom Finley, Lixin Gong, Scott Inglis, Pei Jiang, Ivan Matantsev, Prabhat Roy, M. Zeeshan Siddiqui, Shouheng Yi, Shauheen Zahirazami, Yiwen Zhu.

<!--- SPDX-License-Identifier: Apache-2.0 -->
#

![ONNXMLTools_logo_main](https://raw.githubusercontent.com/onnx/onnxmltools/main/docs/ONNXMLTools_logo_main.png)

[![CI](https://github.com/onnx/onnxmltools/actions/workflows/ci.yml/badge.svg)](https://github.com/onnx/onnxmltools/actions/workflows/ci.yml)

## Introduction

ONNXMLTools converts models from machine learning toolkits into [ONNX](https://onnx.ai). It supports:

* LightGBM
* XGBoost
* CatBoost
* Apple Core ML
* H2O (MOJO models)
* libsvm
* Spark ML (experimental)
* scikit-learn, as a wrapper of [skl2onnx](https://github.com/onnx/sklearn-onnx/)
* TensorFlow and Keras, as a wrapper of [tf2onnx](https://github.com/onnx/tensorflow-onnx/).
  New projects should use tf2onnx directly.

PyTorch has a built-in ONNX exporter; see the [PyTorch documentation](https://pytorch.org/docs/stable/onnx.html).

## Install

Install the latest release from [PyPI](https://pypi.org/project/onnxmltools/):

```bash
pip install onnxmltools
```

or the development version from source:

```bash
pip install git+https://github.com/onnx/onnxmltools
```

## Requirements

ONNXMLTools requires Python **3.11+**, ONNX, NumPy and protobuf.
To convert a model, the toolkit that produced it must also be installed.
CI tests every pull request on Linux, macOS and Windows against the oldest and
newest versions below; the [CI matrix](.github/workflows/ci.yml) lists the exact combinations.

| Package | Tested versions |
|---|---|
| onnx | 1.14.1 – 1.23.2 |
| onnxruntime (tests only) | 1.17.3 – 1.30.0 |
| scikit-learn | 1.4.2 – latest |
| LightGBM | 3.3 – latest 4.x |
| XGBoost | 1.7 – latest 3.x |
| CatBoost, H2O, libsvm, pyspark | latest |
| coremltools | latest (not tested on Windows) |

## Converting a model

Every converter returns an `onnx.ModelProto`.
`initial_types` describes the model inputs: one `(name, type)` pair per input,
with `None` for a dynamic dimension such as the batch size.

### LightGBM

```python
import numpy as np
import onnxruntime as rt
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris

from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

X, y = load_iris(return_X_y=True)
X = X.astype(np.float32)
model = LGBMClassifier(n_estimators=10).fit(X, y)

onnx_model = convert_lightgbm(
    model, initial_types=[("input", FloatTensorType([None, X.shape[1]]))]
)

sess = rt.InferenceSession(
    onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
)
labels, probabilities = sess.run(None, {"input": X[:5]})
```

### XGBoost

```python
import numpy as np
from sklearn.datasets import load_iris
from xgboost import XGBClassifier

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

X, y = load_iris(return_X_y=True)
model = XGBClassifier(n_estimators=10).fit(X.astype(np.float32), y)

onnx_model = convert_xgboost(
    model, initial_types=[("input", FloatTensorType([None, X.shape[1]]))]
)
```

### Core ML

```python
import coremltools
import onnxmltools

coreml_model = coremltools.utils.load_spec("example.mlmodel")
onnx_model = onnxmltools.convert_coreml(coreml_model, "Example Model")
onnxmltools.utils.save_model(onnx_model, "example.onnx")
```

### H2O

The only prerequisite is a MOJO model saved on the local file system.

```python
import onnxmltools

onnx_model = onnxmltools.convert_h2o("/path/to/h2o/gbm_mojo.zip")
onnxmltools.utils.save_model(onnx_model, "h2o_gbm.onnx")
```

More examples, including libsvm, Spark ML and scikit-learn, are in
[docs/examples](docs/examples).

## Choosing the opset

The converters support the default ONNX domain up to opset 15, and target
opset 15 unless the installed `onnx` package only supports an older one. To
produce a model for an older runtime, pass a lower `target_opset`, for example
`convert_lightgbm(model, initial_types=..., target_opset=13)`; a value above 15
raises an error. The [versioning documentation](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions)
maps opsets to ONNX releases.

The opset recorded in the converted model can be lower than `target_opset`:

```python
print(onnx_model.opset_import)
```

Each operator is converted with the newest version available up to
`target_opset`, and the model records the highest opset among the operators it
uses. A LightGBM classifier, for example, records opset 9 even with
`target_opset=15`. This is intended and keeps the model loadable by older
runtimes.

## Running the tests

The tests convert models and compare the predictions of the original model
with those of [onnxruntime](https://pypi.org/project/onnxruntime/).

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
python -m pytest tests/lightgbm tests/xgboost   # or any folder under tests/
```

Each folder under `tests/` needs the matching toolkit installed.

## Adding a converter

A new converter comes with unit tests under `tests/<toolkit>/` that train a
small model, convert it, and compare the original predictions with the
onnxruntime predictions on the same inputs.

## License

[Apache License v2.0](LICENSE)

<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNXMLTools

![ONNXMLTools logo](https://raw.githubusercontent.com/onnx/onnxmltools/main/docs/ONNXMLTools_logo_main.png)

[![CI](https://github.com/onnx/onnxmltools/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/onnx/onnxmltools/actions/workflows/ci.yml)
[![CodeQL](https://github.com/onnx/onnxmltools/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/onnx/onnxmltools/actions/workflows/codeql.yml)
[![PyPI](https://img.shields.io/pypi/v/onnxmltools.svg)](https://pypi.org/project/onnxmltools/)
[![Python](https://img.shields.io/pypi/pyversions/onnxmltools.svg)](https://pypi.org/project/onnxmltools/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Convert trained machine learning models to [ONNX](https://onnx.ai) for portable
inference with runtimes such as [ONNX Runtime](https://onnxruntime.ai).

[Install](#install) · [Quick start](#quick-start) ·
[Supported toolkits](#supported-toolkits) · [Compatibility](#compatibility) ·
[Contributing](#contributing) · [Release notes](CHANGELOGS.md)

## Install

Install the latest release from [PyPI](https://pypi.org/project/onnxmltools/):

```bash
python -m pip install onnxmltools
```

Install the toolkit that produced your model separately. ONNX Runtime is only
needed if you want to run the exported model. For the quick start below:

```bash
python -m pip install onnxmltools lightgbm scikit-learn onnxruntime
```

Install the development version directly from GitHub with:

```bash
python -m pip install git+https://github.com/onnx/onnxmltools.git
```

The development version requires **Python 3.11 or later**. Published releases
may have different requirements; check the Python badge or the release metadata
on PyPI when installing an older version.

## Quick start

A converter returns an `onnx.ModelProto`. The `initial_types` argument describes
each input's name, element type, and shape. In this example, `None` allows a
variable batch size and `FloatTensorType` requires `float32` inputs.

```python
import numpy as np
import onnx
import onnxruntime as rt
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris

from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

X, y = load_iris(return_X_y=True)
X = X.astype(np.float32)
model = LGBMClassifier(
    n_estimators=10, random_state=0, n_jobs=1, verbosity=-1
).fit(X, y)

onnx_model = convert_lightgbm(
    model,
    initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
    target_opset=15,
    zipmap=False,
)
onnx.checker.check_model(onnx_model)
onnx.save_model(onnx_model, "lightgbm.onnx")

session = rt.InferenceSession(
    onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
)
sample = X[:1]
labels, probabilities = session.run(None, {"input": sample})

np.testing.assert_array_equal(labels, model.predict(sample))
np.testing.assert_allclose(
    probabilities, model.predict_proba(sample), rtol=1e-5, atol=1e-6
)
```

For LightGBM classifiers, `zipmap=False` returns probabilities as an array with
one column per class. The default, `zipmap=True`, returns a sequence of
dictionaries that map class labels to probabilities.

`onnx.checker` validates the model's ONNX structure; it does not establish
prediction equivalence. Before deployment, compare the converted model with the
original on representative data, including missing values and categorical
features when applicable. Keep feature order, preprocessing, and input types
consistent with training.

## Supported toolkits

Support depends on a model's operators and configuration, not only its toolkit.

| Toolkit | Converter | Notes |
| --- | --- | --- |
| LightGBM | `convert_lightgbm` | Classifiers, regressors, rankers, and supported Boosters. |
| XGBoost | `convert_xgboost` | Supported classifiers, regressors, and Boosters. |
| CatBoost | `convert_catboost` | Uses CatBoost's ONNX export support. |
| Apple Core ML | `convert_coreml` | Supports selected Core ML model types and operators. |
| H2O | `convert_h2o` | Converts supported GBM MOJO models. |
| libsvm | `onnxmltools.convert.convert_libsvm` | Classification and regression. |
| Spark ML | `convert_sparkml` | Experimental; see the [Spark ML guide](onnxmltools/convert/sparkml/README.md). |
| scikit-learn | `convert_sklearn` | Wraps [skl2onnx](https://github.com/onnx/sklearn-onnx/). |
| TensorFlow / Keras | `convert_tensorflow` / `convert_keras` | Wraps [tf2onnx](https://github.com/onnx/tensorflow-onnx/); use tf2onnx directly for new integrations. |

PyTorch provides its own [ONNX exporter](https://pytorch.org/docs/stable/onnx.html).
More conversion examples are available in [docs/examples](docs/examples).

## Compatibility

The development version requires Python 3.11 or later, ONNX 1.14.1 or later,
and skl2onnx 1.19.1 or later. The [CI matrix](.github/workflows/ci.yml) tests
Python 3.11 through 3.13 on Linux, macOS, and Windows against both minimum and
recent dependency versions. It is the source of truth for exact tested
combinations.

The converters implemented directly in this repository support the default ONNX
domain through **opset 15**. Their default target is 15, limited by the installed
ONNX package. Wrapper converters follow the capabilities of the exporter they
delegate to.

To target an older runtime, choose a supported lower `target_opset` during
conversion and test the result with that runtime. Three version values matter:

- **ONNX package version:** the installed Python library.
- **Opset imports:** operator versions for each domain, such as `ai.onnx.ml`.
- **IR version:** the model format version, which the runtime must support.

Inspect every domain import instead of assuming the first one is the default
domain:

```python
print("IR version:", onnx_model.ir_version)
print({op.domain or "ai.onnx": op.version for op in onnx_model.opset_import})
```

The recorded opset can be lower than the requested target because the converter
records the versions required by the operators it emits. See
[ONNX versioning](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions)
for the relationship between ONNX releases, opsets, and IR versions.

## Contributing

For a focused LightGBM and XGBoost development environment:

```bash
python -m pip install -e .
python -m pip install pytest pandas lightgbm xgboost onnxruntime
python -m pytest tests/lightgbm tests/xgboost
```

The complete development dependency list is in `requirements-dev.txt`.
Individual suites need their matching toolkit and sometimes platform
dependencies such as Java for Spark ML and H2O. Use the
[CI workflow](.github/workflows/ci.yml) as the reference for full environments.

A new converter should include tests under `tests/<toolkit>/` that train a small
model, convert it, and compare its outputs with ONNX Runtime. For a bug report,
include a minimal reproducer, package versions, `target_opset`, the full error,
and expected versus actual outputs. Report issues on
[GitHub](https://github.com/onnx/onnxmltools/issues).

## License

ONNXMLTools is distributed under the [Apache License 2.0](LICENSE).

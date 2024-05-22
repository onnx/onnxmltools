# Change Logs

## 1.13.0 (development)

* Handle issue with binary classifier setting output to [N,1] vs [N,2],
  [#681](https://github.com/onnx/onnxmltools/pull/681)
* Add missing dependency onnxconverter_common, fix multi regression with xgboost,
  [#679](https://github.com/onnx/onnxmltools/pull/679),
  fixes issues [No module named 'onnxconverter_common'](https://github.com/onnx/onnxmltools/issues/673),
  [onnx converted : xgboostRegressor multioutput model predicts 1 dimension instead of original 210 dimensions.](https://github.com/onnx/onnxmltools/issues/676)

## 1.12.0

* Fix early stopping for XGBClassifier and xgboost > 2
  [#597](https://github.com/onnx/onnxmltools/pull/597)
* Fix discrepancies with XGBRegressor and xgboost > 2
  [#670](https://github.com/onnx/onnxmltools/pull/670)
* Support count:poisson for XGBRegressor
  [#666](https://github.com/onnx/onnxmltools/pull/666)
* Supports XGBRFClassifier and XGBRFRegressor
  [#665](https://github.com/onnx/onnxmltools/pull/665)
* ONNX_DFS_PATH to be set in the spark config
  [#653](https://github.com/onnx/onnxmltools/pull/653)
  (by @Ironwood-Cyber)
* Sparkml converter: support type StringType and StringType()
  [#639](https://github.com/onnx/onnxmltools/pull/639)
* Add check for base_score in _get_attributes function
  [#637](https://github.com/onnx/onnxmltools/pull/637),
  [#626](https://github.com/onnx/onnxmltools/pull/626),
  (by @tolleybot)
* Support for lightgbm >= 4.0
  [#634](https://github.com/onnx/onnxmltools/pull/634)

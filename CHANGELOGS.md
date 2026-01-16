# Change Logs

## 1.16.0

* Improve xgboost categorical feature support
  [#743](https://github.com/onnx/onnxmltools/pull/743)

## 1.15.0

* Add support for gamma and tweedie distributions for xgboost
  [#742](https://github.com/onnx/onnxmltools/pull/742)
* Support xgboost 3, including for multiclass problems
  [#736](https://github.com/onnx/onnxmltools/pull/736)
* Add support to convert xgboost models with categorical features
  [#734](https://github.com/onnx/onnxmltools/pull/734)

## 1.14.0

* Add tweedie objective to LightGBM options
  [#722](https://github.com/onnx/onnxmltools/pull/722)
* Support for "huber" objective in the LGBM Booster
  [#705](https://github.com/onnx/onnxmltools/pull/705)
* Remove import of split_complex_to_pairs and unused functions
  [#714](https://github.com/onnx/onnxmltools/pull/714)
* Removes dependency on onnxconveter-common
  [#718](https://github.com/onnx/onnxmltools/pull/718)

## 1.13.0

* Handle issue with binary classifier setting output to [N,1] vs [N,2],
  [#681](https://github.com/onnx/onnxmltools/pull/681)
* Fix multi regression with xgboost,
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

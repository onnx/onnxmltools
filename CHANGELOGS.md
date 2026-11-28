# Change Logs

## 1.17.0

* Deprecate xgboost 1.x and lightgbm 3.x; `convert_xgboost` and
  `convert_lightgbm` now emit a `FutureWarning` for them, and onnxmltools
  1.18.0 will require xgboost 2.0 and lightgbm 4.0 or later
* Require Python 3.11 or later; Python 3.9 and 3.10 are no longer supported
  [#829](https://github.com/onnx/onnxmltools/pull/829)
* Require onnx 1.14.1 or later (previously 1.8.1) and skl2onnx 1.19.1 or later
  (previously 1.4.9), the oldest versions tested
* Fix LightGBM thresholds rounded to float32 above the float64 value,
  which sent one float32 input per threshold to the other branch
  [#806](https://github.com/onnx/onnxmltools/issues/806)
* Fix LightGBM random forests (`boosting_type="rf"`), which were converted
  with wrong predictions
  [#801](https://github.com/onnx/onnxmltools/pull/801)
* Fix LightGBM classifier probabilities for a `sigmoid` parameter other than 1
  and for the `multiclassova` objective
  [#802](https://github.com/onnx/onnxmltools/pull/802)
* Send zeros to the default child of LightGBM splits trained with
  `zero_as_missing=True`
  [#814](https://github.com/onnx/onnxmltools/pull/814)
* Support the `fair`, `mape`, `cross_entropy` and `cross_entropy_lambda`
  LightGBM objectives
  [#804](https://github.com/onnx/onnxmltools/pull/804)
* Raise an error for LightGBM models with linear trees (`linear_tree=True`)
  instead of converting them with wrong predictions
  [#803](https://github.com/onnx/onnxmltools/pull/803)
* Fix `XGBRegressor` with `binary:logistic` giving wrong predictions when the
  trees are not all zero, for example with `subsample < 1.0`
  [#771](https://github.com/onnx/onnxmltools/pull/771)
* Apply the sigmoid for `XGBRegressor` with `reg:logistic`
  [#805](https://github.com/onnx/onnxmltools/pull/805)
* Apply the DART tree weights when converting XGBoost models trained with dropout
  [#813](https://github.com/onnx/onnxmltools/pull/813)
* Fix the exponential conversion time of XGBoost models with native categorical
  splits
  [#793](https://github.com/onnx/onnxmltools/pull/793)
* Add type hints to the `convert_*` functions, accept skl2onnx data types in
  `initial_types` of the scikit-learn wrapper, and fix Core ML converter bugs
  [#776](https://github.com/onnx/onnxmltools/pull/776)
* Declare `packaging` as a dependency
  [#800](https://github.com/onnx/onnxmltools/pull/800)

## 1.16.0

* Add LGBMRanker support
  initial PR [754](https://github.com/onnx/onnxmltools/pull/754),
  finalized in PR [755](https://github.com/onnx/onnxmltools/pull/755)
* Add partial support for custom objective
  [#753](https://github.com/onnx/onnxmltools/pull/753)
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

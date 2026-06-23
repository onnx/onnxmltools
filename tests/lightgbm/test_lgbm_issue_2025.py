import unittest
import numpy as np


class TestLightGBMIssue2025(unittest.TestCase):
    def test_issue_708(self):
        # https://github.com/onnx/onnxmltools/issues/708

        import pprint
        from datetime import datetime, timedelta
        import pandas as pd
        from lightgbm import LGBMRegressor
        import onnx
        import onnxmltools
        import onnxruntime
        from skl2onnx.common.data_types import FloatTensorType

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        date_range = pd.date_range(start=start_date, end=end_date, freq="5min")
        df_timestamps = pd.DataFrame(index=date_range)
        N = len(df_timestamps)

        used = pd.Series([0] * N, index=date_range)
        used[(used.index.dayofweek <= 4) & (used.index.hour == 8)] = 1
        used[(used.index.dayofweek <= 4) & (used.index.hour == 12)] = 2
        used[(used.index.dayofweek <= 4) & (used.index.hour == 14)] = 3

        y = pd.DataFrame(
            {
                "y": used,
            },
            index=date_range,
        )
        X = pd.DataFrame(
            {
                "sin_day_of_week": np.sin(2 * np.pi * date_range.dayofweek / 7),
                "cos_day_of_week": np.cos(2 * np.pi * date_range.dayofweek / 7),
                "sin_hour_of_day": np.sin(2 * np.pi * date_range.hour / 24),
                "cos_hour_of_day": np.cos(2 * np.pi * date_range.hour / 24),
            },
            index=date_range,
        )
        X.columns = [f"f{i}" for i in range(X.shape[1])]

        lgb_model = LGBMRegressor(
            objective="quantile",  # Use quantile loss
            alpha=0.95,  # Quantile for the loss (default is median: 0.5)
            n_estimators=1,  # Number of boosting iterations
            max_depth=2,  # Maximum tree depth
        )
        lgb_model.fit(X, y)

        init_types = [("float_input", FloatTensorType([None, X.shape[1]]))]

        onnx_model_lgmb = onnxmltools.convert_lightgbm(
            lgb_model, initial_types=init_types
        )
        onnx.save(onnx_model_lgmb, "test_issue_708.onnx")

        lgb_predictions = lgb_model.predict(X)

        lgbm_sess = onnxruntime.InferenceSession(
            onnx_model_lgmb.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        loaded_lgb_predictions = lgbm_sess.run(
            output_names=["variable"],
            input_feed={"float_input": X.to_numpy().astype(np.float32)},
        )[0]
        disc = []
        for i, (features, x, y) in enumerate(
            zip(
                X.values.astype(np.float32),
                lgb_predictions,
                loaded_lgb_predictions.ravel(),
            )
        ):
            if abs(x - y) > 1e-5:
                disc.append((i, features, x, np.float32(x), y))
        assert not disc, f"Discrepancies: {pprint.pformat(disc)}"


if __name__ == "__main__":
    unittest.main(verbosity=2)

import pandas as pd

from maintainx.features.rolling import add_rolling_features
from maintainx.features.selection import drop_constant_columns
from maintainx.features.target import cap_rul
from maintainx.features.temporal import add_delta_features


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 2, 1, 2],
            "cycle": [1, 1, 2, 2],
            "sensor_1": [10.0, 100.0, 20.0, 120.0],
            "constant": [1, 1, 1, 1],
        }
    )


def test_drop_constant_columns_removes_only_constant_requested_columns():
    result = drop_constant_columns(_fixture(), ["sensor_1", "constant"])

    assert "constant" not in result.columns
    assert "sensor_1" in result.columns


def test_grouped_features_are_independent_of_input_order():
    frame = _fixture().drop(columns="constant")
    shuffled = frame.sample(frac=1, random_state=7)
    ordered_features = add_delta_features(add_rolling_features(frame, window=2))
    shuffled_features = add_delta_features(add_rolling_features(shuffled, window=2))
    ordered_features = ordered_features.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    shuffled_features = shuffled_features.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    pd.testing.assert_series_equal(
        ordered_features["sensor_1_rolling_mean"],
        shuffled_features["sensor_1_rolling_mean"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        ordered_features["sensor_1_delta"],
        shuffled_features["sensor_1_delta"],
        check_names=False,
    )


def test_first_cycle_delta_is_zero_and_rul_is_capped():
    frame = _fixture().drop(columns="constant")
    deltas = add_delta_features(frame)
    assert (deltas.loc[deltas["cycle"] == 1, "sensor_1_delta"] == 0).all()

    capped = cap_rul(pd.DataFrame({"rul": [10, 130]}), cap=125)
    assert capped["rul"].tolist() == [10, 125]
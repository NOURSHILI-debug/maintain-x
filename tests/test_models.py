"""Tests for maintainx.models.utils."""

import pandas as pd
import pytest

from maintainx.models.utils import split_by_unit


@pytest.fixture
def sample_features_df():
    # 5 units, 4 cycles each — enough to exercise a real split
    rows = []
    for unit_id in range(1, 6):
        for cycle in range(1, 5):
            rows.append({"unit_id": unit_id, "cycle": cycle, "rul": 50 - cycle})
    return pd.DataFrame(rows)


class TestSplitByUnit:
    def test_no_unit_id_overlap_between_train_and_val(self, sample_features_df):
        train_df, val_df = split_by_unit(sample_features_df, val_fraction=0.2, seed=42)
        train_units = set(train_df["unit_id"])
        val_units = set(val_df["unit_id"])
        assert train_units.isdisjoint(val_units)

    def test_all_units_accounted_for(self, sample_features_df):
        train_df, val_df = split_by_unit(sample_features_df, val_fraction=0.2, seed=42)
        all_units = set(sample_features_df["unit_id"])
        split_units = set(train_df["unit_id"]) | set(val_df["unit_id"])
        assert all_units == split_units

    def test_val_fraction_roughly_respected(self, sample_features_df):
        train_df, val_df = split_by_unit(sample_features_df, val_fraction=0.2, seed=42)
        val_unit_count = val_df["unit_id"].nunique()
        # 20% of 5 units = 1 unit
        assert val_unit_count == 1

    def test_reproducible_with_same_seed(self, sample_features_df):
        train_1, val_1 = split_by_unit(sample_features_df, seed=42)
        train_2, val_2 = split_by_unit(sample_features_df, seed=42)
        assert set(val_1["unit_id"]) == set(val_2["unit_id"])

    def test_different_seeds_can_produce_different_splits(self, sample_features_df):
        # Not a strict guarantee for every possible seed pair, but with 5 units
        # and these two seeds this should hold — flags if seed is silently ignored.
        _, val_a = split_by_unit(sample_features_df, seed=1)
        _, val_b = split_by_unit(sample_features_df, seed=2)
        assert set(val_a["unit_id"]) != set(val_b["unit_id"])
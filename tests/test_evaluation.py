"""Tests for maintainx.evaluation.metrics."""

import pytest

from maintainx.evaluation.metrics import (
    compute_rmse,
    compute_mae,
    compute_nasa_score,
    evaluate_model,
)


class TestComputeRMSE:
    def test_perfect_predictions_zero_rmse(self):
        y_true = [50, 30, 10]
        y_pred = [50, 30, 10]
        assert compute_rmse(y_true, y_pred) == 0

    def test_rmse_positive_for_imperfect_predictions(self):
        y_true = [50, 30, 10]
        y_pred = [45, 35, 8]
        assert compute_rmse(y_true, y_pred) > 0


class TestComputeMAE:
    def test_perfect_predictions_zero_mae(self):
        y_true = [50, 30, 10]
        y_pred = [50, 30, 10]
        assert compute_mae(y_true, y_pred) == 0

    def test_mae_matches_known_value(self):
        # errors: |5|, |5| -> mean = 5
        y_true = [50, 30]
        y_pred = [55, 25]
        assert compute_mae(y_true, y_pred) == pytest.approx(5.0)


class TestComputeNasaScore:
    def test_perfect_predictions_zero_score(self):
        y_true = [50, 30, 10]
        y_pred = [50, 30, 10]
        assert compute_nasa_score(y_true, y_pred) == pytest.approx(0.0)

    def test_late_prediction_penalized_more_than_early(self):
        # Same magnitude of error, opposite direction.
        # Late (predicted RUL too high) must score worse than
        # early (predicted RUL too low) — this is the core asymmetry
        # the NASA score exists to encode.
        y_true = [50]
        late_pred = [60]   # d = +10, late
        early_pred = [40]  # d = -10, early

        late_score = compute_nasa_score(y_true, late_pred)
        early_score = compute_nasa_score(y_true, early_pred)

        assert late_score > early_score

    def test_score_increases_with_error_magnitude(self):
        y_true = [50]
        small_error_pred = [55]
        large_error_pred = [70]

        small_score = compute_nasa_score(y_true, small_error_pred)
        large_score = compute_nasa_score(y_true, large_error_pred)

        assert large_score > small_score

    def test_score_sums_across_samples(self):
        # Score for two samples should equal the sum of their individual scores
        y_true = [50, 30]
        y_pred = [55, 35]

        combined_score = compute_nasa_score(y_true, y_pred)
        individual_sum = (
            compute_nasa_score([50], [55]) + compute_nasa_score([30], [35])
        )

        assert combined_score == pytest.approx(individual_sum)


class TestEvaluateModel:
    def test_returns_expected_keys(self):
        y_true = [50, 30, 10]
        y_pred = [48, 32, 9]

        result = evaluate_model(y_true, y_pred)

        assert set(result.keys()) == {"rmse", "mae", "nasa_score"}

    def test_values_are_floats(self):
        y_true = [50, 30, 10]
        y_pred = [48, 32, 9]

        result = evaluate_model(y_true, y_pred)

        assert isinstance(result["rmse"], float)
        assert isinstance(result["mae"], float)
        assert isinstance(result["nasa_score"], float)
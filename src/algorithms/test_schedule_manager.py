
import pytest

from src.algorithms.schedule_manager import ScheduleManager


def test_linear_schedule_reaches_start_and_end_values():
    schedule = ScheduleManager(
        strategy="linear",
        gamma_start=0.90,
        gamma_end=0.99,
        lambda_start=0.80,
        lambda_end=0.95,
        total_updates=5,
    )

    assert schedule.get_values(1) == pytest.approx((0.90, 0.80))
    assert schedule.get_values(5) == pytest.approx((0.99, 0.95))


def test_linear_schedule_interpolates_intermediate_values():
    schedule = ScheduleManager(
        strategy="linear",
        gamma_start=0.90,
        gamma_end=0.99,
        lambda_start=0.80,
        lambda_end=0.95,
        total_updates=5,
    )

    gamma, gae_lambda = schedule.get_values(3)

    assert gamma == pytest.approx(0.945)
    assert gae_lambda == pytest.approx(0.875)


def test_fixed_schedule_keeps_parameters_constant():
    schedule = ScheduleManager(
        strategy="fixed",
        gamma_start=0.99,
        gamma_end=0.90,
        lambda_start=0.95,
        lambda_end=0.80,
        total_updates=5,
    )

    for update in range(1, 6):
        assert schedule.get_values(update) == (0.99, 0.95)


def test_single_update_linear_schedule_uses_end_values():
    schedule = ScheduleManager(
        strategy="linear",
        gamma_start=0.90,
        gamma_end=0.99,
        lambda_start=0.80,
        lambda_end=0.95,
        total_updates=1,
    )

    assert schedule.get_values(1) == (0.99, 0.95)


def test_invalid_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy"):
        ScheduleManager(strategy="exponential")


def test_invalid_update_boundaries_are_rejected():
    schedule = ScheduleManager(total_updates=5)

    with pytest.raises(ValueError, match="update"):
        schedule.get_values(0)

    with pytest.raises(ValueError, match="update"):
        schedule.get_values(6)


def test_invalid_total_updates_are_rejected():
    with pytest.raises(ValueError, match="total_updates"):
        ScheduleManager(total_updates=0)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("gamma_start", -0.1),
        ("gamma_end", 1.1),
        ("lambda_start", -0.1),
        ("lambda_end", 1.1),
    ],
)
def test_invalid_schedule_parameters_are_rejected(parameter, value):
    with pytest.raises(ValueError):
        ScheduleManager(**{parameter: value})

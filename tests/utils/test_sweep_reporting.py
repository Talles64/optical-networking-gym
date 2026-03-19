from __future__ import annotations

from optical_networking_gym_v2.utils.sweep_reporting import aggregate_summary_metrics


def test_aggregate_summary_metrics_keeps_only_requested_kpis() -> None:
    rows = [
        {
            "services_accepted": 10.0,
            "services_served": 9.0,
            "service_blocking_rate": 0.1,
            "bit_rate_blocking_rate": 0.2,
            "mean_osnr_final": 17.0,
            "blocked_due_to_resources": 2.0,
        },
        {
            "services_accepted": 14.0,
            "services_served": 12.0,
            "service_blocking_rate": 0.2,
            "bit_rate_blocking_rate": 0.3,
            "mean_osnr_final": 19.0,
            "blocked_due_to_resources": 5.0,
        },
    ]

    aggregated = aggregate_summary_metrics(
        rows,
        metric_names=(
            "services_accepted",
            "services_served",
            "service_blocking_rate",
            "bit_rate_blocking_rate",
            "mean_osnr_final",
        ),
    )

    assert aggregated["services_accepted_mean"] == 12.0
    assert aggregated["services_accepted_std"] == 2.0
    assert aggregated["mean_osnr_final_mean"] == 18.0
    assert "blocked_due_to_resources_mean" not in aggregated

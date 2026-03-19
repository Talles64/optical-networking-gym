from __future__ import annotations

import statistics
from typing import Callable

from optical_networking_gym_v2 import get_modulations, select_first_fit_action


DEFAULT_MODULATION_NAMES = "BPSK,QPSK,8QAM,16QAM,32QAM,64QAM"

EpisodePolicy = Callable[[object, dict[str, object]], int]


def float_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def float_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def build_modulation_index_to_name(modulation_names: str) -> dict[int, str]:
    return {
        int(modulation.spectral_efficiency): f"modulation_{modulation.name.lower()}".replace("-", "_")
        for modulation in get_modulations(modulation_names)
    }


def select_masked_first_fit_policy(env: object, info: dict[str, object]) -> int:
    mask = info.get("mask")
    if mask is None and hasattr(env, "action_masks"):
        mask = env.action_masks()
    if mask is None:
        raise RuntimeError("first-fit policy requires an action mask")
    return int(select_first_fit_action(mask))


def episode_modulation_counts(
    statistics_snapshot,
    modulation_index_to_name: dict[int, str],
) -> dict[str, float]:
    counts_by_key = {column_name: 0.0 for column_name in modulation_index_to_name.values()}
    for spectral_efficiency, count in statistics_snapshot.episode_modulation_histogram:
        column_name = modulation_index_to_name.get(int(spectral_efficiency))
        if column_name is None:
            continue
        counts_by_key[column_name] = float(count)
    return counts_by_key


__all__ = [
    "DEFAULT_MODULATION_NAMES",
    "EpisodePolicy",
    "build_modulation_index_to_name",
    "episode_modulation_counts",
    "float_mean",
    "float_std",
    "select_masked_first_fit_policy",
]

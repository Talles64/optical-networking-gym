from __future__ import annotations

import math

import numpy as np


_ABS_BETA_2 = abs(-21.3e-27)
_GAMMA = 1.3e-3
_H_PLANCK = 6.626e-34
_PI_SQUARED = math.pi * math.pi
_NLI_PREFACTOR_BASE = 8.0 / (27.0 * math.pi * _ABS_BETA_2)


def accumulate_link_noise(
    span_lengths_km: np.ndarray,
    span_attenuation_normalized: np.ndarray,
    span_noise_figure_normalized: np.ndarray,
    running_service_ids: np.ndarray,
    running_center_frequencies: np.ndarray,
    running_bandwidths: np.ndarray,
    running_phi_modulation: np.ndarray,
    *,
    current_service_id: int,
    center_frequency: float,
    bandwidth: float,
    launch_power: float,
    include_nli: bool,
) -> tuple[float, float, float]:
    lengths = np.asarray(span_lengths_km, dtype=np.float64)
    attenuations = np.asarray(span_attenuation_normalized, dtype=np.float64)
    noise_figures = np.asarray(span_noise_figure_normalized, dtype=np.float64)
    service_ids = np.asarray(running_service_ids, dtype=np.int32)
    center_frequencies = np.asarray(running_center_frequencies, dtype=np.float64)
    running_bandwidth_values = np.asarray(running_bandwidths, dtype=np.float64)
    phi_modulation = np.asarray(running_phi_modulation, dtype=np.float64)

    acc_gsnr = 0.0
    acc_ase = 0.0
    acc_nli = 0.0
    nli_prefactor = ((launch_power / bandwidth) ** 3) * _NLI_PREFACTOR_BASE * (_GAMMA**2) * bandwidth

    for span_index, span_length_km in enumerate(lengths):
        span_length_m = span_length_km * 1e3
        attenuation = attenuations[span_index]
        power_nli_span = 0.0

        if include_nli:
            l_eff_a = 1.0 / (2.0 * attenuation)
            l_eff = (1.0 - math.exp(-2.0 * attenuation * span_length_m)) / (2.0 * attenuation)
            sum_phi = math.asinh(_PI_SQUARED * _ABS_BETA_2 * (bandwidth**2) / (4.0 * attenuation))

            for running_index, running_service_id in enumerate(service_ids):
                if running_service_id == current_service_id:
                    continue
                delta_frequency = center_frequencies[running_index] - center_frequency
                if delta_frequency == 0.0:
                    continue
                running_bandwidth = running_bandwidth_values[running_index]
                phi = (
                    math.asinh(
                        _PI_SQUARED
                        * _ABS_BETA_2
                        * l_eff_a
                        * running_bandwidth
                        * (delta_frequency + (running_bandwidth / 2.0))
                    )
                    - math.asinh(
                        _PI_SQUARED
                        * _ABS_BETA_2
                        * l_eff_a
                        * running_bandwidth
                        * (delta_frequency - (running_bandwidth / 2.0))
                    )
                ) - (
                    phi_modulation[running_index]
                    * (running_bandwidth / abs(delta_frequency))
                    * (5.0 / 3.0)
                    * (l_eff / span_length_m)
                )
                sum_phi += phi

            power_nli_span = nli_prefactor * l_eff * sum_phi

        power_ase = (
            bandwidth
            * _H_PLANCK
            * center_frequency
            * (math.exp(2.0 * attenuation * span_length_m) - 1.0)
            * noise_figures[span_index]
        )

        if include_nli:
            acc_gsnr += (power_ase + power_nli_span) / launch_power
            if power_nli_span > 0.0:
                acc_nli += power_nli_span / launch_power
        else:
            acc_gsnr += power_ase / launch_power

        acc_ase += power_ase / launch_power

    return float(acc_gsnr), float(acc_ase), float(acc_nli)


__all__ = ["accumulate_link_noise"]

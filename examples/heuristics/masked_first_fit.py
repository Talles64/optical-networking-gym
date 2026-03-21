from __future__ import annotations

from pathlib import Path

from optical_networking_gym_v2 import BUILTIN_TOPOLOGY_DIR, make_env, set_topology_dir
from optical_networking_gym_v2.heuristics.masked_heuristics import select_first_fit_action


TOPOLOGY_DIR = BUILTIN_TOPOLOGY_DIR


def run_episode(seed: int = 7) -> dict[str, float | int | str]:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="nobel-eu",
        modulation_names="QPSK,BPSK,8QAM,16QAM,32QAM,64QAM",
        topology_dir=TOPOLOGY_DIR,
        seed=seed,
        bit_rates=(10, 40, 100, 400),
        load=350.0,
        mean_holding_time=10800,
        num_spectrum_resources=320,
        episode_length=1000,
        modulations_to_consider=2,
        k_paths=2,
    )
    _, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    last_status = "unknown"

    while True:
        mask = info.get("mask")
        if mask is None:
            mask = env.action_masks()
        action = select_first_fit_action(mask)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        last_status = str(info.get("status", "unknown"))
        if terminated or truncated:
            break

    return {
        "mode": "masked",
        "policy": "first_fit",
        "steps": steps,
        "total_reward": total_reward,
        "last_status": last_status,
        "Info": info,
    }


def main() -> None:
    print(run_episode())


if __name__ == "__main__":
    main()

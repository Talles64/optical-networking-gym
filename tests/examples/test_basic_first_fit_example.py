from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from optical_networking_gym_v2 import set_topology_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_ROOT.parent / "examples" / "topologies"
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "basic_first_fit.py"


def test_basic_first_fit_example_writes_results_file(tmp_path: Path) -> None:
    set_topology_dir(TOPOLOGY_DIR)
    module = _load_example_module()

    summary = module.run_episode(topology_name="ring_4", seed=7)
    output_path = tmp_path / "basic_first_fit.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["topology_name"] == "ring_4"
    assert payload["steps"] == 100
    assert "episode_service_blocking_rate" in payload


def _load_example_module():
    spec = importlib.util.spec_from_file_location("basic_first_fit_example", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load example module at {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

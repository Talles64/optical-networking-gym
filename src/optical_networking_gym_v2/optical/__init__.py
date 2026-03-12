from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "QoTEngine": (".qot_engine", "QoTEngine"),
    "accumulate_link_noise": (".kernels", "accumulate_link_noise"),
    "block_is_free": (".kernels", "block_is_free"),
    "candidate_starts_array": (".kernels", "candidate_starts_array"),
    "fill_range": (".kernels", "fill_range"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

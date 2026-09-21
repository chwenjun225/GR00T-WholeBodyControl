"""Low-overhead timing aggregation for the Isaac Sim control loop."""

from __future__ import annotations


class TimingProfiler:
    def __init__(self) -> None:
        self._samples: dict[str, list[int]] = {}

    def add(self, name: str, elapsed_ns: int) -> None:
        sample = self._samples.get(name)
        if sample is None:
            self._samples[name] = [elapsed_ns, elapsed_ns, 1]
            return
        sample[0] += elapsed_ns
        sample[1] = max(sample[1], elapsed_ns)
        sample[2] += 1

    def format(self, names: tuple[str, ...]) -> str:
        fields = []
        for name in names:
            total_ns, maximum_ns, count = self._samples.get(name, (0, 0, 0))
            average_ms = total_ns / max(count, 1) / 1e6
            maximum_ms = maximum_ns / 1e6
            fields.append(f"{name}={average_ms:.3f}/{maximum_ms:.3f}ms x{count}")
        return " ".join(fields)

    def reset(self) -> None:
        self._samples.clear()

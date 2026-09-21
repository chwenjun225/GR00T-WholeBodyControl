"""Batched read-only state access backed by Isaac Sim's C++ prim-data views."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np


def _host_float_array(view, getter_name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Wrap a prim-data host pointer as a persistent, zero-copy NumPy view."""
    expected = int(np.prod(shape))
    result = getattr(view, getter_name)()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"{getter_name} returned an invalid pointer/count pair: {result!r}")

    first, second = (int(result[0]), int(result[1]))
    if second == expected:
        pointer, count = first, second
    elif first == expected:  # tolerate bindings exposing (count, pointer)
        count, pointer = first, second
    else:
        raise RuntimeError(
            f"{getter_name} returned {result!r}; expected {expected} float values"
        )
    if pointer == 0 or count != expected:
        raise RuntimeError(f"{getter_name} returned an empty host buffer")

    buffer = (ctypes.c_float * count).from_address(pointer)
    return np.ctypeslib.as_array(buffer).reshape(shape)


@dataclass(frozen=True)
class BatchedRobotState:
    positions: np.ndarray
    orientations: np.ndarray
    linear_velocities: np.ndarray
    angular_velocities: np.ndarray


class CppStateReader:
    """Persistent NumPy views over batched C++ host buffers.

    The rigid-body paths must be ordered as pelvis, torso.
    """

    def __init__(self, rigid_bodies):
        rigid_bodies.initialize_cpp_data_view()
        self._rigid_bodies = rigid_bodies
        self._rigid_body_view = rigid_bodies._cpp_data_view
        if self._rigid_body_view is None:
            raise RuntimeError("Isaac C++ rigid-body data view is unavailable")

        # Host getters lazily register and allocate their provider buffers.
        # Calling update() before this registration crashes the 6.0.1 PhysX
        # prim-data provider, so bind all requested fields first.
        self._state = BatchedRobotState(
            positions=_host_float_array(
                self._rigid_body_view, "get_world_positions_host", (2, 3)
            ),
            orientations=_host_float_array(
                self._rigid_body_view, "get_world_orientations_host", (2, 4)
            ),
            linear_velocities=_host_float_array(
                self._rigid_body_view, "get_linear_velocities_host", (2, 3)
            ),
            angular_velocities=_host_float_array(
                self._rigid_body_view, "get_angular_velocities_host", (2, 3)
            ),
        )
        self._update_views()

    def _update_views(self) -> None:
        if not self._rigid_body_view.update():
            raise RuntimeError("RigidBodyDataView.update() failed")

    def read(self) -> BatchedRobotState:
        self._update_views()
        return self._state

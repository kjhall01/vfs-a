"""Shared explicit Runge–Kutta sequences."""
from .explicit import ssprk3, forward_euler, rk4

__all__ = ["ssprk3", "forward_euler", "rk4"]

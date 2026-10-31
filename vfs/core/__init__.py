"""Compatibility surface; core implementations contain only shared contracts."""
__all__ = ["State", "PrognosticFields", "StateSpec", "TracerSpec", "ssprk3", "forward_euler", "rk4"]


def __getattr__(name):
    if name in ("State", "PrognosticFields", "StateSpec", "TracerSpec"):
        from vfs.atmosphere import state
        return getattr(state, name)
    if name in ("ssprk3", "forward_euler", "rk4"):
        from . import integrators
        return getattr(integrators, name)
    raise AttributeError(name)

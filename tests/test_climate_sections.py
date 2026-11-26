"""Check the visual encoding and public notebook calls on a small analytic section."""
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("matplotlib").use("Agg")
from matplotlib import pyplot as plt

from vfs_plotting import plot_section, plot_sections, plot_zonal_mean


@pytest.fixture
def summary():
    latitude = np.linspace(-75, 75, 16)
    pressure = np.geomspace(1, 1000, 12)
    wind = np.broadcast_to(latitude[:, None], (16, 12)).copy()
    overturning = np.broadcast_to(np.log10(pressure), wind.shape).copy() - 1.5
    return xr.Dataset(
        {"temperature": (("lat", "pressure_hpa"), 260. + wind / 10 + overturning),
         "zonal_wind": (("lat", "pressure_hpa"), wind),
         "streamfunction": (("lat", "pressure_hpa"), overturning),
         "sst_zonal": ("lat", 30. - abs(latitude) / 3)},
        coords={"lat": latitude, "pressure_hpa": pressure}, attrs={"period": "Test state"},
    )


@pytest.mark.parametrize("field,axis", [("zonal_wind", 0), ("streamfunction", 1)])
def test_overlay_on_user_axes(summary, field, axis):
    fig, ax = plt.subplots()
    assert plot_section(summary, field, ax=ax, levels=[-1, 0, 1]) is ax
    assert ax.get_yscale() == "log"
    assert ax.get_ylim() == (1000, 1)
    np.testing.assert_allclose(ax.collections[0].get_array().reshape(12, 16), summary.temperature.T)
    contour = ax.collections[1]
    np.testing.assert_array_equal(contour.levels, [-1, 0, 1])
    # The zero contour is at 0° latitude for u and sqrt(1000) hPa for psi.
    crossing = contour.allsegs[1][0][:, axis]
    if field == "zonal_wind":
        np.testing.assert_allclose(crossing, 0, atol=1e-10)
    else:
        np.testing.assert_allclose(crossing, crossing[0])
        assert 30 < crossing[0] < 35
    fig.canvas.draw()
    plt.close(fig)


def test_linear_pair_and_individual_zonal_profile(summary):
    fig = plot_sections(summary, pressure_scale="linear")
    assert len(fig.axes) == 3
    assert all(ax.get_yscale() == "linear" for ax in fig.axes[:2])
    for ax in fig.axes[:2]:
        np.testing.assert_allclose(ax.collections[0].get_clim(), [summary.temperature.min(), summary.temperature.max()])
    plt.close(fig)
    ax = plot_zonal_mean(summary, "sst")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), summary.sst_zonal)
    plt.close(ax.figure)

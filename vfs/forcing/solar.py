"""Torch-native solar forcing with diurnal, seasonal, and orbital controls."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..core.registry import register_component


class OrbitalSolarForcing(nn.Module):
    """Compute per-cell top-of-atmosphere insolation.

    This is a compact astronomical forcing, not a full ephemeris.  It is built
    for climate workflow forcing: tensor-native, differentiable with respect to
    latitude/longitude, and explicit about the orbital parameters being used.
    ``orbital_year_offset`` lets long integrations or sensitivity tests sample
    slow Milankovitch-style changes without making those cycles implicit.
    """

    recipe_name = "orbital_solar_forcing"

    def __init__(
        self,
        *,
        mode: str = "diurnal_seasonal",
        solar_constant: float = 1361.0,
        year_days: float = 365.2422,
        day_seconds: float = 86400.0,
        start_day_of_year: float = 80.0,
        vernal_equinox_day: float = 80.0,
        daily_mean: bool = False,
        obliquity_deg: float = 23.4392911,
        eccentricity: float = 0.0167086,
        longitude_perihelion_deg: float = 282.9372,
        orbital_year_offset: float = 0.0,
        long_term_cycles: bool = True,
        obliquity_period_years: float = 41000.0,
        obliquity_amplitude_deg: float = 1.3,
        eccentricity_period_years: float = 100000.0,
        eccentricity_amplitude: float = 0.012,
        axial_precession_period_years: float = 25772.0,
        apsidal_precession_period_years: float = 112000.0,
        min_zenith_factor: float = 0.0,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        if self.mode not in ("diurnal_seasonal", "orbital"):
            raise ValueError("OrbitalSolarForcing mode must be 'diurnal_seasonal' or 'orbital'")
        self.daily_mean = bool(daily_mean)
        self.long_term_cycles = bool(long_term_cycles)
        self.solar_constant = float(solar_constant)
        self.year_days = float(year_days)
        self.day_seconds = float(day_seconds)
        self.start_day_of_year = float(start_day_of_year)
        self.vernal_equinox_day = float(vernal_equinox_day)
        self.obliquity_deg = float(obliquity_deg)
        self.eccentricity = float(eccentricity)
        self.longitude_perihelion_deg = float(longitude_perihelion_deg)
        self.orbital_year_offset = float(orbital_year_offset)
        self.obliquity_period_years = float(obliquity_period_years)
        self.obliquity_amplitude_deg = float(obliquity_amplitude_deg)
        self.eccentricity_period_years = float(eccentricity_period_years)
        self.eccentricity_amplitude = float(eccentricity_amplitude)
        self.axial_precession_period_years = float(axial_precession_period_years)
        self.apsidal_precession_period_years = float(apsidal_precession_period_years)
        self.min_zenith_factor = float(min_zenith_factor)
        if self.solar_constant <= 0.0:
            raise ValueError("solar_constant must be positive")
        if self.year_days <= 0.0 or self.day_seconds <= 0.0:
            raise ValueError("year_days and day_seconds must be positive")

    def _scalar(self, value: float, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)

    def _orbital_parameters(self, years: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        two_pi = self._scalar(2.0 * math.pi, ref)
        obliquity = self._scalar(math.radians(self.obliquity_deg), ref)
        eccentricity = self._scalar(self.eccentricity, ref)
        perihelion = self._scalar(math.radians(self.longitude_perihelion_deg), ref)
        if self.mode == "orbital" and self.long_term_cycles:
            obliquity = obliquity + self._scalar(math.radians(self.obliquity_amplitude_deg), ref) * torch.sin(
                two_pi * years / self._scalar(self.obliquity_period_years, ref)
            )
            eccentricity = (
                eccentricity
                + self._scalar(self.eccentricity_amplitude, ref)
                * torch.sin(two_pi * years / self._scalar(self.eccentricity_period_years, ref))
            ).clamp(0.0, 0.08)
            precession = years * (
                1.0 / self._scalar(self.axial_precession_period_years, ref)
                + 1.0 / self._scalar(self.apsidal_precession_period_years, ref)
            )
            perihelion = perihelion + two_pi * precession
        return obliquity, eccentricity, perihelion

    def forward(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        time_seconds: torch.Tensor | float,
        *,
        surface_albedo: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        lat = torch.as_tensor(lat)
        lon = torch.as_tensor(lon, dtype=lat.dtype, device=lat.device)
        time = torch.as_tensor(time_seconds, dtype=lat.dtype, device=lat.device)
        two_pi = torch.as_tensor(2.0 * math.pi, dtype=lat.dtype, device=lat.device)
        day = time / self._scalar(self.day_seconds, lat)
        years = self._scalar(self.orbital_year_offset, lat) + day / self._scalar(self.year_days, lat)

        solar_longitude = two_pi * (
            (day + self._scalar(self.start_day_of_year - self.vernal_equinox_day, lat))
            / self._scalar(self.year_days, lat)
        )
        obliquity, eccentricity, perihelion = self._orbital_parameters(years, lat)
        declination = torch.asin(torch.sin(obliquity) * torch.sin(solar_longitude))
        true_anomaly = solar_longitude - perihelion
        distance_factor = ((1.0 + eccentricity * torch.cos(true_anomaly)) / (1.0 - eccentricity.square())).square()

        sin_lat = torch.sin(lat)
        cos_lat = torch.cos(lat)
        sin_dec = torch.sin(declination)
        cos_dec = torch.cos(declination)
        if self.daily_mean:
            x = (-torch.tan(lat) * torch.tan(declination)).clamp(-1.0, 1.0)
            half_day = torch.acos(x)
            polar_day = -torch.tan(lat) * torch.tan(declination) <= -1.0
            polar_night = -torch.tan(lat) * torch.tan(declination) >= 1.0
            half_day = torch.where(polar_day, torch.full_like(half_day, math.pi), half_day)
            half_day = torch.where(polar_night, torch.zeros_like(half_day), half_day)
            cos_zenith = (half_day * sin_lat * sin_dec + cos_lat * cos_dec * torch.sin(half_day)) / math.pi
        else:
            hour_angle = lon + two_pi * torch.remainder(day, self._scalar(1.0, lat))
            cos_zenith = sin_lat * sin_dec + cos_lat * cos_dec * torch.cos(hour_angle)
        zenith_factor = (distance_factor * cos_zenith).clamp_min(float(self.min_zenith_factor))
        toa = self._scalar(self.solar_constant, lat) * zenith_factor
        albedo = (
            torch.full_like(toa, 0.30)
            if surface_albedo is None
            else torch.as_tensor(surface_albedo, dtype=toa.dtype, device=toa.device)
        )
        absorbed_surface = toa * (1.0 - albedo.clamp(0.0, 1.0))
        return {
            "solar_constant": torch.full_like(toa, float(self.solar_constant)),
            "zenith_factor": zenith_factor,
            "solar_zenith_factor": zenith_factor,
            "toa_insolation": toa,
            "solar_toa_insolation": toa,
            "absorbed_shortwave_surface_down": absorbed_surface,
            "solar_absorbed_shortwave_surface_down": absorbed_surface,
            "solar_cos_zenith": cos_zenith,
            "solar_declination_rad": torch.broadcast_to(declination, toa.shape),
            "solar_earth_sun_distance_factor": torch.broadcast_to(distance_factor, toa.shape),
        }


register_component("orbital_solar_forcing", OrbitalSolarForcing)


__all__ = ["OrbitalSolarForcing"]

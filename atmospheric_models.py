import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime
import logging

class NRLMSISE00:
    """
    NRLMSISE-00 atmospheric density model implementation
    with space weather integration for accurate drag modeling
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NRLMSISE00, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize NRLMSISE-00 model"""
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            
            # Model constants
            self.R_earth = 6371.0  # km
            self.scale_height_base = 8.5  # km
            
            # Default space weather parameters
            self.default_f107 = 150.0  # Solar flux (10.7 cm)
            self.default_kp = 3.0      # Geomagnetic index
            
            self.logger.debug("NRLMSISE-00 atmospheric model initialized")
            NRLMSISE00._initialized = True
    
    def get_density(self, altitude_km: float, latitude: float, longitude: float, 
                   datetime_utc: datetime, f107: float = None, 
                   kp: float = None) -> float:
        """
        Get atmospheric density at specified conditions
        
        Args:
            altitude_km: Altitude in kilometers
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            datetime_utc: UTC datetime
            f107: Solar 10.7 cm flux (optional)
            kp: Geomagnetic index (optional)
            
        Returns:
            Atmospheric density in kg/m^3
        """
        try:
            # Use defaults if space weather not provided
            if f107 is None:
                f107 = self.default_f107
            if kp is None:
                kp = self.default_kp
            
            # For this implementation, we'll use a simplified NRLMSISE-00 model
            # A full implementation would require the complete NRLMSISE-00 Fortran code
            # or a Python wrapper like nrlmsise00
            
            # Base exponential atmosphere
            base_density = self._exponential_atmosphere(altitude_km)
            
            # Apply space weather corrections
            solar_correction = self._solar_flux_correction(f107, altitude_km)
            geomagnetic_correction = self._geomagnetic_correction(kp, altitude_km)
            
            # Diurnal and seasonal variations
            diurnal_correction = self._diurnal_variation(
                longitude, datetime_utc, altitude_km
            )
            seasonal_correction = self._seasonal_variation(
                latitude, datetime_utc, altitude_km
            )
            
            # Combine all corrections
            density = (base_density * 
                      solar_correction * 
                      geomagnetic_correction * 
                      diurnal_correction * 
                      seasonal_correction)
            
            return max(density, 1e-15)  # Minimum density threshold
            
        except Exception as e:
            self.logger.error(f"Density calculation error: {e}")
            return self._fallback_density(altitude_km)
    
    def _exponential_atmosphere(self, altitude_km: float) -> float:
        """Base exponential atmosphere model"""
        if altitude_km < 0:
            altitude_km = 0
        
        # Sea level density
        rho_0 = 1.225  # kg/m^3
        
        # Altitude-dependent scale height
        if altitude_km < 100:
            H = 8.5  # km
        elif altitude_km < 200:
            H = 27.0  # km
        elif altitude_km < 300:
            H = 60.0  # km
        elif altitude_km < 500:
            H = 100.0  # km
        else:
            H = 150.0  # km
        
        # Exponential decay
        density = rho_0 * np.exp(-altitude_km / H)
        
        return density
    
    def _solar_flux_correction(self, f107: float, altitude_km: float) -> float:
        """Solar flux correction factor"""
        # Normalized solar flux (relative to quiet conditions)
        f107_norm = f107 / 150.0
        
        # Altitude-dependent solar influence
        if altitude_km < 200:
            solar_influence = 0.1
        elif altitude_km < 400:
            solar_influence = 0.5
        else:
            solar_influence = 1.0
        
        # Correction factor
        correction = 1.0 + solar_influence * (f107_norm - 1.0)
        
        return max(correction, 0.1)
    
    def _geomagnetic_correction(self, kp: float, altitude_km: float) -> float:
        """Geomagnetic activity correction factor"""
        # Normalized Kp index
        kp_norm = kp / 3.0
        
        # Altitude-dependent geomagnetic influence
        if altitude_km < 300:
            geo_influence = 0.05
        elif altitude_km < 500:
            geo_influence = 0.2
        else:
            geo_influence = 0.4
        
        # Correction factor (geomagnetic heating increases density)
        correction = 1.0 + geo_influence * (kp_norm - 1.0)
        
        return max(correction, 0.5)
    
    def _diurnal_variation(self, longitude: float, datetime_utc: datetime, 
                          altitude_km: float) -> float:
        """Diurnal (daily) density variation"""
        # Local solar time calculation
        hour_utc = datetime_utc.hour + datetime_utc.minute / 60.0
        hour_local = hour_utc + longitude / 15.0
        hour_local = hour_local % 24
        
        # Diurnal amplitude depends on altitude
        if altitude_km < 200:
            amplitude = 0.1
        elif altitude_km < 400:
            amplitude = 0.3
        else:
            amplitude = 0.5
        
        # Peak density around 14:00 local time
        phase = 2 * np.pi * (hour_local - 14.0) / 24.0
        correction = 1.0 + amplitude * np.cos(phase)
        
        return correction
    
    def _seasonal_variation(self, latitude: float, datetime_utc: datetime, 
                           altitude_km: float) -> float:
        """Seasonal density variation"""
        # Day of year
        day_of_year = datetime_utc.timetuple().tm_yday
        
        # Seasonal amplitude depends on latitude and altitude
        lat_factor = abs(np.sin(np.radians(latitude)))
        
        if altitude_km < 200:
            amplitude = 0.05 * lat_factor
        elif altitude_km < 400:
            amplitude = 0.15 * lat_factor
        else:
            amplitude = 0.25 * lat_factor
        
        # Peak density around day 100 (spring in northern hemisphere)
        phase = 2 * np.pi * (day_of_year - 100) / 365.25
        correction = 1.0 + amplitude * np.cos(phase)
        
        return correction
    
    def _fallback_density(self, altitude_km: float) -> float:
        """Fallback density calculation for error cases"""
        return self._exponential_atmosphere(altitude_km)

class ExponentialAtmosphere:
    """
    Simple exponential atmosphere model as backup
    """
    
    def __init__(self):
        """Initialize exponential model"""
        self.logger = logging.getLogger(__name__)
        
    def get_density(self, altitude_km: float, **kwargs) -> float:
        """Get density using simple exponential model"""
        if altitude_km < 0:
            altitude_km = 0
        
        # Sea level density
        rho_0 = 1.225  # kg/m^3
        
        # Scale height
        H = 8.5  # km
        
        # Exponential decay
        density = rho_0 * np.exp(-altitude_km / H)
        
        return density

class JacchiaRoberts:
    """
    Jacchia-Roberts atmospheric model (simplified implementation)
    """
    
    def __init__(self):
        """Initialize Jacchia-Roberts model"""
        self.logger = logging.getLogger(__name__)
        
    def get_density(self, altitude_km: float, f107: float = 150.0, 
                   **kwargs) -> float:
        """Get density using Jacchia-Roberts model"""
        if altitude_km < 90:
            return 0.0
        
        # Temperature calculation
        T_inf = 900 + 2.5 * (f107 - 70)  # Exospheric temperature
        T_120 = T_inf  # Simplified
        
        # Density calculation (simplified)
        if altitude_km < 120:
            # Below 120 km
            rho = 3e-9 * np.exp(-(altitude_km - 120) / 5.0)
        else:
            # Above 120 km
            T = T_120 * (1 - 0.5 * np.exp(-(altitude_km - 120) / 50))
            rho = 6e-10 * (T_120 / T) * np.exp(-(altitude_km - 120) / (T / 28.0))
        
        return rho

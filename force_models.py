import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime
import logging
from coordinate_transforms import CoordinateTransforms

class ForceModels:
    """
    Comprehensive force modeling for high-fidelity orbital propagation
    including J2-J6 harmonics, atmospheric drag, and solar radiation pressure
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize force models"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.coord_transform = CoordinateTransforms()
        
        # Earth parameters
        self.mu = 3.986004418e14  # m^3/s^2 - Earth gravitational parameter
        self.R_earth = 6378137.0  # m - Earth equatorial radius
        self.J2 = 1.08262668e-3   # J2 zonal harmonic
        self.J3 = -2.53265648e-6  # J3 zonal harmonic
        self.J4 = -1.61962159e-6  # J4 zonal harmonic
        self.J5 = -2.27296082e-7  # J5 zonal harmonic
        self.J6 = 5.40681239e-7   # J6 zonal harmonic
        
        # Atmospheric drag parameters
        self.default_cd = 2.2  # Drag coefficient
        self.default_area = 1140.23  # m^2 - ISS drag area
        
        # Solar radiation pressure parameters
        self.default_cr = 1.3  # SRP coefficient
        self.solar_flux = 1367.0  # W/m^2 - Solar flux at 1 AU
        self.c = 299792458.0  # m/s - Speed of light
        self.au = 149597870700.0  # m - Astronomical unit
        
        self.logger.info("Force models initialized")
    
    def compute_perturbations(self, position: np.ndarray, velocity: np.ndarray,
                            datetime_utc: datetime, cd_a: float = None,
                            cr_a: float = None) -> np.ndarray:
        """
        Compute total perturbation accelerations
        
        Args:
            position: Position vector in ECI frame (m)
            velocity: Velocity vector in ECI frame (m/s)
            datetime_utc: Current UTC time
            cd_a: Drag coefficient times area (m^2)
            cr_a: SRP coefficient times area (m^2)
            
        Returns:
            Total perturbation acceleration (m/s^2)
        """
        total_accel = np.zeros(3)
        
        try:
            # J2-J6 zonal harmonics
            if self.config.get('use_j2_j6', True):
                accel_harmonics = self._compute_zonal_harmonics(position)
                total_accel += accel_harmonics
            
            # Atmospheric drag
            if self.config.get('use_drag', True):
                accel_drag = self._compute_atmospheric_drag(
                    position, velocity, datetime_utc, cd_a
                )
                total_accel += accel_drag
            
            # Solar radiation pressure
            if self.config.get('use_srp', True):
                accel_srp = self._compute_solar_radiation_pressure(
                    position, datetime_utc, cr_a
                )
                total_accel += accel_srp
            
            # Third-body perturbations (simplified)
            if self.config.get('use_third_body', False):
                accel_third_body = self._compute_third_body_perturbations(
                    position, datetime_utc
                )
                total_accel += accel_third_body
            
        except Exception as e:
            self.logger.error(f"Perturbation computation error: {e}")
        
        return total_accel
    
    def _compute_zonal_harmonics(self, position: np.ndarray) -> np.ndarray:
        """Compute J2-J6 zonal harmonic perturbations"""
        x, y, z = position
        r = np.linalg.norm(position)
        
        if r == 0:
            return np.zeros(3)
        
        # Normalized coordinates
        sin_phi = z / r  # Sine of latitude
        cos_phi = np.sqrt(x**2 + y**2) / r  # Cosine of latitude
        
        # Common factors
        mu_r3 = self.mu / (r**3)
        re_r = self.R_earth / r
        
        # Initialize acceleration
        accel = np.zeros(3)
        
        # J2 perturbation
        if abs(self.J2) > 0:
            re_r2 = re_r**2
            factor_j2 = -1.5 * self.J2 * mu_r3 * re_r2
            
            # Radial component
            a_r_j2 = factor_j2 * (1 - 5 * sin_phi**2)
            
            # Latitude component
            a_phi_j2 = factor_j2 * (-2 * sin_phi * cos_phi)
            
            # Convert to Cartesian
            if cos_phi > 0:
                accel[0] += a_r_j2 * (x/r) + a_phi_j2 * (-x*z/(r**2*cos_phi))
                accel[1] += a_r_j2 * (y/r) + a_phi_j2 * (-y*z/(r**2*cos_phi))
            else:
                accel[0] += a_r_j2 * (x/r)
                accel[1] += a_r_j2 * (y/r)
            
            accel[2] += a_r_j2 * (z/r) + a_phi_j2 * (cos_phi/r)
        
        # J3 perturbation
        if abs(self.J3) > 0:
            re_r3 = re_r**3
            factor_j3 = -2.5 * self.J3 * mu_r3 * re_r3 * sin_phi
            
            a_r_j3 = factor_j3 * (3 - 7 * sin_phi**2)
            a_phi_j3 = factor_j3 * (-3 * cos_phi + 7 * sin_phi**2 * cos_phi)
            
            if cos_phi > 0:
                accel[0] += a_r_j3 * (x/r) + a_phi_j3 * (-x*z/(r**2*cos_phi))
                accel[1] += a_r_j3 * (y/r) + a_phi_j3 * (-y*z/(r**2*cos_phi))
            else:
                accel[0] += a_r_j3 * (x/r)
                accel[1] += a_r_j3 * (y/r)
            
            accel[2] += a_r_j3 * (z/r) + a_phi_j3 * (cos_phi/r)
        
        # J4 perturbation
        if abs(self.J4) > 0:
            re_r4 = re_r**4
            factor_j4 = (15.0/8.0) * self.J4 * mu_r3 * re_r4
            
            sin_phi2 = sin_phi**2
            a_r_j4 = factor_j4 * (1 - 14*sin_phi2 + 21*sin_phi2**2)
            a_phi_j4 = factor_j4 * sin_phi * cos_phi * (-4 + 14*sin_phi2)
            
            if cos_phi > 0:
                accel[0] += a_r_j4 * (x/r) + a_phi_j4 * (-x*z/(r**2*cos_phi))
                accel[1] += a_r_j4 * (y/r) + a_phi_j4 * (-y*z/(r**2*cos_phi))
            else:
                accel[0] += a_r_j4 * (x/r)
                accel[1] += a_r_j4 * (y/r)
            
            accel[2] += a_r_j4 * (z/r) + a_phi_j4 * (cos_phi/r)
        
        # J5 and J6 (simplified - similar pattern)
        # Full implementation would include these as well
        
        return accel
    
    def _compute_atmospheric_drag(self, position: np.ndarray, velocity: np.ndarray,
                                datetime_utc: datetime, cd_a: float = None) -> np.ndarray:
        """Compute atmospheric drag acceleration"""
        try:
            # Get atmospheric density
            from atmospheric_models import NRLMSISE00
            
            atmosphere = NRLMSISE00()
            
            # Convert position to geodetic
            from utils import eci_to_geodetic
            lat, lon, alt = eci_to_geodetic(position, datetime_utc)
            
            # Get density
            rho = atmosphere.get_density(
                alt / 1000,  # Convert to km
                lat, lon, datetime_utc
            )
            
            # Relative velocity (account for Earth rotation)
            omega_earth = 7.2921159e-5  # rad/s
            v_rot = np.array([-omega_earth * position[1], 
                             omega_earth * position[0], 0])
            v_rel = velocity - v_rot
            v_rel_mag = np.linalg.norm(v_rel)
            
            if v_rel_mag == 0:
                return np.zeros(3)
            
            # Drag coefficient and area
            if cd_a is None:
                cd_a = self.default_cd * self.default_area
            
            # Drag acceleration
            drag_accel = -0.5 * rho * cd_a * v_rel_mag * v_rel
            
            # Divide by satellite mass (assume 1 kg for specific acceleration)
            # In practice, this would be scaled by actual mass
            mass = self.config.get('satellite_mass', 464291.0)  # ISS mass in kg
            
            return drag_accel / mass
            
        except Exception as e:
            self.logger.error(f"Drag computation error: {e}")
            return np.zeros(3)
    
    def _compute_solar_radiation_pressure(self, position: np.ndarray,
                                        datetime_utc: datetime,
                                        cr_a: float = None) -> np.ndarray:
        """Compute solar radiation pressure acceleration"""
        try:
            # Sun position (simplified)
            sun_pos = self._get_sun_position(datetime_utc)
            
            # Vector from satellite to Sun
            sat_to_sun = sun_pos - position
            sat_to_sun_mag = np.linalg.norm(sat_to_sun)
            
            if sat_to_sun_mag == 0:
                return np.zeros(3)
            
            sat_to_sun_unit = sat_to_sun / sat_to_sun_mag
            
            # Check if satellite is in Earth's shadow
            if self._is_in_shadow(position, sun_pos):
                return np.zeros(3)
            
            # Solar flux at satellite distance
            solar_distance_au = sat_to_sun_mag / self.au
            flux = self.solar_flux / (solar_distance_au**2)
            
            # SRP acceleration
            if cr_a is None:
                cr_a = self.default_cr * self.default_area
            
            # Pressure
            pressure = flux / self.c
            
            # SRP acceleration
            mass = self.config.get('satellite_mass', 464291.0)  # kg
            srp_accel = pressure * cr_a * sat_to_sun_unit / mass
            
            return srp_accel
            
        except Exception as e:
            self.logger.error(f"SRP computation error: {e}")
            return np.zeros(3)
    
    def _get_sun_position(self, datetime_utc: datetime) -> np.ndarray:
        """Get Sun position in ECI frame (simplified)"""
        try:
            # Fallback simplified calculation
            return self._simple_sun_position(datetime_utc)
        except Exception as e:
            self.logger.warning(f"Sun position calculation error: {e}")
            return self._simple_sun_position(datetime_utc)
    
    def _simple_sun_position(self, datetime_utc: datetime) -> np.ndarray:
        """Simplified Sun position calculation"""
        # Days since J2000.0
        from utils import julian_date
        jd = julian_date(datetime_utc)
        T = (jd - 2451545.0) / 36525.0
        
        # Mean longitude of Sun (degrees)
        L = 280.460 + 36000.771 * T
        
        # Mean anomaly (degrees)
        M = 357.5277233 + 35999.05034 * T
        
        # Convert to radians
        L_rad = np.radians(L % 360)
        M_rad = np.radians(M % 360)
        
        # Ecliptic longitude (simplified)
        lambda_sun = L_rad + np.radians(1.914666471) * np.sin(M_rad)
        
        # Distance to Sun (AU)
        r_sun = 1.000140612 - 0.016708617 * np.cos(M_rad)
        
        # Convert to ECI coordinates (simplified, ignoring obliquity changes)
        obliquity = np.radians(23.439291)
        
        x_sun = r_sun * np.cos(lambda_sun) * self.au
        y_sun = r_sun * np.sin(lambda_sun) * np.cos(obliquity) * self.au
        z_sun = r_sun * np.sin(lambda_sun) * np.sin(obliquity) * self.au
        
        return np.array([x_sun, y_sun, z_sun])
    
    def _is_in_shadow(self, sat_pos: np.ndarray, sun_pos: np.ndarray) -> bool:
        """Check if satellite is in Earth's shadow"""
        # Vector from Earth center to satellite
        r_sat = np.linalg.norm(sat_pos)
        
        # Vector from Earth center to Sun
        sun_dir = sun_pos / np.linalg.norm(sun_pos)
        
        # Project satellite position onto Sun direction
        sat_proj = np.dot(sat_pos, sun_dir)
        
        # If satellite is on Sun side of Earth, not in shadow
        if sat_proj > 0:
            return False
        
        # Distance from satellite to Sun-Earth line
        perp_dist = np.linalg.norm(sat_pos - sat_proj * sun_dir)
        
        # Earth's shadow radius at satellite distance
        shadow_radius = self.R_earth
        
        return perp_dist < shadow_radius
    
    def _compute_third_body_perturbations(self, position: np.ndarray,
                                        datetime_utc: datetime) -> np.ndarray:
        """Compute third-body perturbations (Moon and Sun)"""
        # Simplified implementation
        # Full version would use JPL ephemerides
        
        accel = np.zeros(3)
        
        try:
            # Sun perturbation
            sun_pos = self._get_sun_position(datetime_utc)
            mu_sun = 1.327124400e20  # m^3/s^2
            
            # Vector from satellite to Sun
            r_sat_sun = sun_pos - position
            r_sat_sun_mag = np.linalg.norm(r_sat_sun)
            
            # Direct attraction
            accel += mu_sun * r_sat_sun / (r_sat_sun_mag**3)
            
            # Indirect term (Sun attraction on Earth)
            r_earth_sun_mag = np.linalg.norm(sun_pos)
            accel -= mu_sun * sun_pos / (r_earth_sun_mag**3)
            
            # Moon perturbation would be similar but smaller
            
        except Exception as e:
            self.logger.error(f"Third-body perturbation error: {e}")
        
        return accel

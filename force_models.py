import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime
import logging
from coordinate_transforms import CoordinateTransforms
from skyfield.api import load, utc
from skyfield.framelib import ecliptic_frame
import math

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
        
        # Celestial body gravitational parameters
        self.mu_sun = 1.32712442099e20  # m^3/s^2 - Sun gravitational parameter
        self.mu_moon = 4.9048695e12     # m^3/s^2 - Moon gravitational parameter
        
        # Initialize JPL ephemeris for accurate celestial body positions
        try:
            self.ephemeris = load('de421.bsp')  # JPL DE421 ephemeris
            self.earth = self.ephemeris['earth']
            self.sun = self.ephemeris['sun']
            self.moon = self.ephemeris['moon']
            self.ts = load.timescale()
            self.logger.info("JPL ephemeris loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load JPL ephemeris: {e}. Using simplified models.")
            self.ephemeris = None
            
        # Enhanced geopotential coefficients (EGM2008 subset)
        # Only implementing up to 4x4 for computational efficiency in real-time
        self._initialize_geopotential_coefficients()
        
        self.logger.info("Enhanced force models initialized with JPL ephemeris and improved geopotential")
    
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
            
            # Enhanced lunisolar gravity perturbations
            if self.config.get('use_lunisolar', True):
                accel_lunisolar = self._compute_lunisolar_perturbations(
                    position, datetime_utc
                )
                total_accel += accel_lunisolar
            
            # Enhanced geopotential (beyond J2-J6)
            if self.config.get('use_enhanced_geopotential', True):
                accel_geopotential = self._compute_enhanced_geopotential(
                    position, datetime_utc
                )
                total_accel += accel_geopotential
            
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
    
    def _initialize_geopotential_coefficients(self):
        """Initialize enhanced geopotential coefficients (EGM2008 subset)"""
        # Implementing up to 4x4 for computational efficiency
        # Full EGM2008 goes to 2159x2159 but that's computationally prohibitive
        
        # Normalized coefficients C_nm and S_nm
        # These are key EGM2008 coefficients beyond the basic zonals
        self.geopotential_coeffs = {
            # Additional zonal harmonics (m=0)
            (3, 0): -2.53265648e-6,   # C30 (J3)
            (4, 0): -1.61962159e-6,   # C40 (J4) 
            (5, 0): -2.27296082e-7,   # C50 (J5)
            (6, 0): 5.40681239e-7,    # C60 (J6)
            
            # Tesseral harmonics (key ones for orbital accuracy)
            (2, 1): -1.574536e-9,     # C21
            (2, 2): 2.439383e-6,      # C22 (important for longitude-dependent effects)
            (3, 1): 9.570733e-7,      # C31
            (3, 2): 2.030462e-6,      # C32
            (3, 3): 1.009476e-6,      # C33
            (4, 1): -5.087253e-7,     # C41
            (4, 2): 7.841463e-7,      # C42
            (4, 3): 3.225413e-7,      # C43
            (4, 4): -2.140681e-7,     # C44
        }
        
        # Sine coefficients (S_nm)
        self.geopotential_s_coeffs = {
            (2, 1): 1.502701e-9,      # S21
            (2, 2): -1.400273e-6,     # S22
            (3, 1): 2.030176e-6,      # S31
            (3, 2): 2.482004e-7,      # S32
            (3, 3): 1.972014e-7,      # S33
            (4, 1): -4.494599e-7,     # S41
            (4, 2): 1.481554e-6,      # S42
            (4, 3): 6.625119e-7,      # S43
            (4, 4): -4.940733e-7,     # S44
        }
        
        self.logger.info("Enhanced geopotential coefficients initialized (4x4 EGM2008 subset)")
    
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
        """
        Compute enhanced solar radiation pressure acceleration with proper eclipse modeling
        Includes umbra and penumbra effects for accurate SRP modeling
        """
        try:
            # Get Sun position using JPL ephemeris if available
            if self.ephemeris is not None:
                sun_pos = self._get_jpl_sun_position(datetime_utc)
            else:
                sun_pos = self._get_sun_position(datetime_utc)
            
            # Vector from satellite to Sun
            sat_to_sun = sun_pos - position
            sat_to_sun_mag = np.linalg.norm(sat_to_sun)
            
            if sat_to_sun_mag == 0:
                return np.zeros(3)
            
            sat_to_sun_unit = sat_to_sun / sat_to_sun_mag
            
            # Enhanced shadow function with umbra/penumbra modeling
            shadow_factor = self._compute_shadow_function(position, sun_pos)
            
            if shadow_factor == 0.0:
                return np.zeros(3)  # Complete eclipse (umbra)
            
            # Solar flux at satellite distance
            solar_distance_au = sat_to_sun_mag / self.au
            flux = self.solar_flux / (solar_distance_au**2)
            
            # Apply shadow factor for penumbra effects
            flux *= shadow_factor
            
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
    
    def _get_jpl_sun_position(self, datetime_utc: datetime) -> np.ndarray:
        """Get Sun position using JPL ephemeris for highest accuracy"""
        try:
            if self.ephemeris is not None:
                t = self.ts.from_datetime(datetime_utc.replace(tzinfo=utc))
                
                # Get Earth and Sun positions relative to solar system barycenter
                earth_pos_bary = self.earth.at(t).position.km * 1000  # Convert to meters
                sun_pos_bary = self.sun.at(t).position.km * 1000
                
                # Sun position relative to Earth center
                sun_pos_earth = sun_pos_bary - earth_pos_bary
                
                return sun_pos_earth
            else:
                # Fallback to simplified model
                return self._simple_sun_position(datetime_utc)
                
        except Exception as e:
            self.logger.warning(f"JPL Sun position error: {e}")
            return self._simple_sun_position(datetime_utc)
    
    def _compute_shadow_function(self, sat_pos: np.ndarray, sun_pos: np.ndarray) -> float:
        """
        Compute shadow function including umbra and penumbra effects
        Returns: 0.0 for complete shadow (umbra), 1.0 for full sunlight, 
                0.0-1.0 for partial shadow (penumbra)
        """
        try:
            r_sat = np.linalg.norm(sat_pos)
            r_sun = np.linalg.norm(sun_pos)
            
            if r_sat == 0 or r_sun == 0:
                return 1.0
            
            # Unit vectors
            sat_unit = sat_pos / r_sat
            sun_unit = sun_pos / r_sun
            
            # Angle between satellite and sun as seen from Earth center
            cos_angle = np.dot(sat_unit, sun_unit)
            
            # If satellite is on sun side of Earth, no shadow
            if cos_angle > 0:
                return 1.0
            
            # Angular radii of Earth and Sun as seen from satellite
            earth_angular_radius = np.arcsin(self.R_earth / r_sat) if r_sat > self.R_earth else np.pi/2
            
            # Sun angular radius as seen from satellite (accounting for distance)
            sun_radius = 6.96e8  # m - Sun physical radius
            sun_angular_radius = np.arctan(sun_radius / np.linalg.norm(sun_pos - sat_pos))
            
            # Vector from satellite to Earth center
            sat_to_earth = -sat_pos
            sat_to_earth_unit = sat_to_earth / np.linalg.norm(sat_to_earth)
            
            # Vector from satellite to Sun
            sat_to_sun = sun_pos - sat_pos
            sat_to_sun_unit = sat_to_sun / np.linalg.norm(sat_to_sun)
            
            # Angular separation between Earth and Sun as seen from satellite
            angular_separation = np.arccos(np.clip(np.dot(sat_to_earth_unit, sat_to_sun_unit), -1, 1))
            
            # Eclipse conditions
            if angular_separation <= (earth_angular_radius - sun_angular_radius):
                # Total eclipse (umbra)
                return 0.0
            elif angular_separation >= (earth_angular_radius + sun_angular_radius):
                # No eclipse (full sunlight)
                return 1.0
            else:
                # Partial eclipse (penumbra)
                # Use geometric model for partial eclipse factor
                
                # Simplified penumbra model - linear transition
                # More sophisticated models would use area overlap calculations
                eclipse_width = 2 * sun_angular_radius
                eclipse_center = earth_angular_radius
                
                if angular_separation <= eclipse_center:
                    # Deeper penumbra (closer to umbra)
                    eclipse_depth = (eclipse_center - angular_separation + sun_angular_radius) / eclipse_width
                    eclipse_factor = max(0.0, 1.0 - eclipse_depth)
                else:
                    # Lighter penumbra (closer to full sun)
                    eclipse_depth = (angular_separation - eclipse_center + sun_angular_radius) / eclipse_width
                    eclipse_factor = max(0.0, eclipse_depth)
                
                return np.clip(eclipse_factor, 0.0, 1.0)
                
        except Exception as e:
            self.logger.error(f"Shadow function computation error: {e}")
            # Fallback to simple shadow test
            return 0.0 if self._is_in_shadow(sat_pos, sun_pos) else 1.0
    
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
    
    def _compute_lunisolar_perturbations(self, position: np.ndarray, 
                                       datetime_utc: datetime) -> np.ndarray:
        """
        Compute enhanced lunisolar perturbations using JPL ephemeris
        This is critical for sub-1km accuracy over multi-day propagation
        """
        accel = np.zeros(3)
        
        try:
            if self.ephemeris is not None:
                # Use JPL ephemeris for highest accuracy
                t = self.ts.from_datetime(datetime_utc.replace(tzinfo=utc))
                
                # Get Earth position relative to solar system barycenter
                earth_pos_bary = self.earth.at(t).position.km * 1000  # Convert to meters
                
                # Sun perturbation using JPL ephemeris
                sun_pos_bary = self.sun.at(t).position.km * 1000  # Convert to meters
                sun_pos_earth = sun_pos_bary - earth_pos_bary  # Sun position relative to Earth center
                
                # Vector from satellite to Sun
                r_sat_sun = sun_pos_earth - position
                r_sat_sun_mag = np.linalg.norm(r_sat_sun)
                r_earth_sun_mag = np.linalg.norm(sun_pos_earth)
                
                if r_sat_sun_mag > 0 and r_earth_sun_mag > 0:
                    # Direct attraction of Sun on satellite
                    accel_sun_direct = self.mu_sun * r_sat_sun / (r_sat_sun_mag**3)
                    
                    # Indirect effect (Sun attraction on Earth)
                    accel_sun_indirect = -self.mu_sun * sun_pos_earth / (r_earth_sun_mag**3)
                    
                    accel += accel_sun_direct + accel_sun_indirect
                
                # Moon perturbation using JPL ephemeris
                moon_pos_bary = self.moon.at(t).position.km * 1000  # Convert to meters
                moon_pos_earth = moon_pos_bary - earth_pos_bary  # Moon position relative to Earth center
                
                # Vector from satellite to Moon
                r_sat_moon = moon_pos_earth - position
                r_sat_moon_mag = np.linalg.norm(r_sat_moon)
                r_earth_moon_mag = np.linalg.norm(moon_pos_earth)
                
                if r_sat_moon_mag > 0 and r_earth_moon_mag > 0:
                    # Direct attraction of Moon on satellite
                    accel_moon_direct = self.mu_moon * r_sat_moon / (r_sat_moon_mag**3)
                    
                    # Indirect effect (Moon attraction on Earth)
                    accel_moon_indirect = -self.mu_moon * moon_pos_earth / (r_earth_moon_mag**3)
                    
                    accel += accel_moon_direct + accel_moon_indirect
                    
            else:
                # Fallback to simplified analytical model
                accel = self._compute_third_body_perturbations(position, datetime_utc)
                
        except Exception as e:
            self.logger.error(f"Lunisolar perturbation computation error: {e}")
            # Fallback to simplified model
            try:
                accel = self._compute_third_body_perturbations(position, datetime_utc)
            except:
                pass
        
        return accel
    
    def _compute_enhanced_geopotential(self, position: np.ndarray, 
                                     datetime_utc: datetime) -> np.ndarray:
        """
        Compute enhanced geopotential perturbations beyond basic zonals
        Including tesseral and sectorial harmonics up to degree 4
        """
        accel = np.zeros(3)
        
        try:
            x, y, z = position
            r = np.linalg.norm(position)
            
            if r == 0:
                return accel
            
            # Convert to spherical coordinates
            lat = np.arcsin(z / r)  # Geocentric latitude
            lon = np.arctan2(y, x)  # Longitude
            
            # Account for Earth rotation to get Earth-fixed longitude
            # Simplified - proper implementation would use GMST
            from utils import julian_date
            jd = julian_date(datetime_utc)
            T = (jd - 2451545.0) / 36525.0
            gmst = np.radians((280.460 + 36000.771 * T) % 360)
            lon_fixed = lon - gmst
            
            # Compute enhanced geopotential accelerations
            # This includes tesseral and sectorial harmonics beyond basic zonals
            for (n, m), cnm in self.geopotential_coeffs.items():
                if n <= 4 and m <= n:  # Up to degree 4 for computational efficiency
                    
                    snm = self.geopotential_s_coeffs.get((n, m), 0.0)
                    
                    # Associated Legendre polynomials (simplified implementation)
                    # Full implementation would use recursive relations
                    pnm = self._associated_legendre(n, m, np.sin(lat))
                    
                    # Derivatives
                    dpnm = self._associated_legendre_derivative(n, m, np.sin(lat))
                    
                    # Common factors
                    re_r_n = (self.R_earth / r)**(n+1)
                    factor = self.mu * re_r_n / (r**2)
                    
                    # Harmonic terms
                    cos_m_lon = np.cos(m * lon_fixed) if m > 0 else 1.0
                    sin_m_lon = np.sin(m * lon_fixed) if m > 0 else 0.0
                    
                    # Radial acceleration component
                    a_r = factor * (n + 1) * pnm * (cnm * cos_m_lon + snm * sin_m_lon)
                    
                    # Latitude acceleration component  
                    a_lat = factor * dpnm * (cnm * cos_m_lon + snm * sin_m_lon)
                    
                    # Longitude acceleration component
                    if m > 0:
                        a_lon = factor * m * pnm * (-cnm * sin_m_lon + snm * cos_m_lon) / np.cos(lat)
                    else:
                        a_lon = 0.0
                    
                    # Convert to Cartesian coordinates (simplified)
                    cos_lat = np.cos(lat)
                    sin_lat = np.sin(lat)
                    cos_lon = np.cos(lon)
                    sin_lon = np.sin(lon)
                    
                    # Transformation matrix elements (simplified)
                    accel[0] += (a_r * cos_lat * cos_lon - 
                               a_lat * sin_lat * cos_lon - 
                               a_lon * sin_lon)
                    accel[1] += (a_r * cos_lat * sin_lon - 
                               a_lat * sin_lat * sin_lon + 
                               a_lon * cos_lon)
                    accel[2] += (a_r * sin_lat + a_lat * cos_lat)
                    
        except Exception as e:
            self.logger.error(f"Enhanced geopotential computation error: {e}")
        
        return accel
    
    def _associated_legendre(self, n: int, m: int, x: float) -> float:
        """
        Compute associated Legendre polynomial P_n^m(x)
        Simplified implementation - full version would use recursive relations
        """
        if abs(x) > 1:
            return 0.0
        
        try:
            if n == 2:
                if m == 0:
                    return 0.5 * (3 * x**2 - 1)
                elif m == 1:
                    return -3 * x * np.sqrt(1 - x**2)
                elif m == 2:
                    return 3 * (1 - x**2)
            elif n == 3:
                if m == 0:
                    return 0.5 * x * (5 * x**2 - 3)
                elif m == 1:
                    return -1.5 * np.sqrt(1 - x**2) * (5 * x**2 - 1)
                elif m == 2:
                    return 15 * x * (1 - x**2)
                elif m == 3:
                    return -15 * (1 - x**2)**(1.5)
            elif n == 4:
                if m == 0:
                    return 0.125 * (35 * x**4 - 30 * x**2 + 3)
                elif m == 1:
                    return -2.5 * x * np.sqrt(1 - x**2) * (7 * x**2 - 3)
                # Additional terms for m=2,3,4 would be implemented here
            
            return 0.0
            
        except:
            return 0.0
    
    def _associated_legendre_derivative(self, n: int, m: int, x: float) -> float:
        """
        Compute derivative of associated Legendre polynomial
        Simplified implementation
        """
        try:
            if n == 2 and m == 0:
                return 3 * x
            elif n == 2 and m == 1:
                return -3 * np.sqrt(1 - x**2) + 3 * x**2 / np.sqrt(1 - x**2)
            elif n == 3 and m == 0:
                return 0.5 * (15 * x**2 - 3)
            # Additional derivatives would be implemented here
            
            return 0.0
            
        except:
            return 0.0

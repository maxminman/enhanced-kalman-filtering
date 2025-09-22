import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
import math

from satellite_characterizer import SatelliteProperties, SatelliteType, OrbitalRegime

class EnhancedSRPModel:
    """
    Enhanced Solar Radiation Pressure model with adaptive coefficients,
    precise eclipse modeling, and satellite-specific corrections
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced SRP model"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Physical constants
        self.solar_flux_1au = 1367.0  # W/m² at 1 AU
        self.c = 299792458.0  # m/s - Speed of light
        self.au = 149597870700.0  # m - Astronomical unit
        self.earth_radius = 6378137.0  # m
        self.sun_radius = 6.96e8  # m
        
        # Earth orbital parameters for seasonal variations
        self.earth_orbit_eccentricity = 0.0167
        
        # Eclipse modeling parameters
        self.use_precise_eclipse = self.config.get('precise_eclipse_modeling', True)
        self.use_penumbra = self.config.get('use_penumbra_modeling', True)
        
        # Satellite-specific modeling
        self.use_adaptive_coefficients = self.config.get('adaptive_srp_coefficients', True)
        self.attitude_modeling = self.config.get('attitude_modeling', 'conservative')
        
        # Initialize JPL ephemeris if available
        self.ephemeris = None
        self.ts = None
        try:
            from skyfield.api import load
            self.ephemeris = load('de421.bsp')
            self.earth = self.ephemeris['earth']
            self.sun = self.ephemeris['sun']
            self.ts = load.timescale()
            self.logger.info("JPL ephemeris loaded for precise SRP modeling")
        except Exception as e:
            self.logger.warning(f"JPL ephemeris not available: {e}. Using analytical models.")
        
        self.logger.info("Enhanced SRP model initialized")
    
    def compute_srp_acceleration(self, satellite_props: SatelliteProperties,
                               position_eci: np.ndarray, velocity_eci: np.ndarray,
                               datetime_utc: datetime) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute enhanced SRP acceleration with satellite-specific adaptations
        
        Args:
            satellite_props: Satellite properties from characterizer
            position_eci: Position in ECI frame (m)
            velocity_eci: Velocity in ECI frame (m/s)
            datetime_utc: Current UTC time
            
        Returns:
            Tuple of (srp_acceleration, diagnostics_dict)
        """
        try:
            # Get Sun position
            sun_pos_eci = self._get_precise_sun_position(datetime_utc)
            
            # Vector from satellite to Sun
            sat_to_sun = sun_pos_eci - position_eci
            sat_to_sun_mag = np.linalg.norm(sat_to_sun)
            
            if sat_to_sun_mag == 0:
                return np.zeros(3), {'error': 'Zero sun distance'}
            
            sat_to_sun_unit = sat_to_sun / sat_to_sun_mag
            
            # Compute eclipse factor
            eclipse_factor, eclipse_info = self._compute_enhanced_eclipse_factor(
                position_eci, sun_pos_eci
            )
            
            if eclipse_factor == 0.0:
                return np.zeros(3), {
                    'eclipse_factor': eclipse_factor,
                    'eclipse_type': eclipse_info.get('type', 'umbra'),
                    'sun_distance_au': sat_to_sun_mag / self.au
                }
            
            # Compute solar flux with seasonal variations
            solar_flux = self._compute_seasonal_solar_flux(datetime_utc, sat_to_sun_mag)
            
            # Apply eclipse factor
            effective_flux = solar_flux * eclipse_factor
            
            # Compute adaptive SRP coefficient
            adaptive_cr = self._compute_adaptive_srp_coefficient(
                satellite_props, position_eci, sat_to_sun_unit
            )
            
            # Compute SRP pressure
            pressure = effective_flux / self.c
            
            # Compute SRP acceleration
            srp_area = satellite_props.srp_area
            mass = satellite_props.mass
            
            # Basic SRP acceleration (assumes perfect reflection)
            srp_accel_basic = pressure * adaptive_cr * srp_area * sat_to_sun_unit / mass
            
            # Apply attitude-dependent corrections
            srp_accel = self._apply_attitude_corrections(
                srp_accel_basic, satellite_props, sat_to_sun_unit, velocity_eci
            )
            
            # Diagnostics
            diagnostics = {
                'eclipse_factor': eclipse_factor,
                'eclipse_type': eclipse_info.get('type', 'sunlight'),
                'solar_flux': solar_flux,
                'effective_flux': effective_flux,
                'adaptive_cr': adaptive_cr,
                'sun_distance_au': sat_to_sun_mag / self.au,
                'srp_pressure': pressure,
                'srp_magnitude': np.linalg.norm(srp_accel)
            }
            
            return srp_accel, diagnostics
            
        except Exception as e:
            self.logger.error(f"Enhanced SRP computation failed: {e}")
            return np.zeros(3), {'error': str(e)}
    
    def _get_precise_sun_position(self, datetime_utc: datetime) -> np.ndarray:
        """Get precise Sun position using JPL ephemeris or analytical model"""
        try:
            if self.ephemeris is not None:
                # Use JPL ephemeris for highest precision
                from skyfield.api import utc
                t = self.ts.from_datetime(datetime_utc.replace(tzinfo=utc))
                
                # Get Earth and Sun positions relative to solar system barycenter
                earth_pos = self.earth.at(t).position.km * 1000  # Convert to meters
                sun_pos = self.sun.at(t).position.km * 1000
                
                # Sun position relative to Earth center
                sun_pos_earth = sun_pos - earth_pos
                
                return sun_pos_earth
            else:
                # Fallback to analytical model
                return self._analytical_sun_position(datetime_utc)
                
        except Exception as e:
            self.logger.warning(f"Precise Sun position calculation failed: {e}")
            return self._analytical_sun_position(datetime_utc)
    
    def _analytical_sun_position(self, datetime_utc: datetime) -> np.ndarray:
        """Analytical Sun position calculation with improved accuracy"""
        try:
            # Days since J2000.0
            from utils import julian_date
            jd = julian_date(datetime_utc)
            T = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000
            
            # Mean longitude of Sun (degrees)
            L = 280.460 + 36000.771 * T
            
            # Mean anomaly (degrees)
            M = 357.5277233 + 35999.05034 * T
            M_rad = np.radians(M % 360)
            
            # Equation of center (higher order terms for better accuracy)
            C = (1.914666471 * np.sin(M_rad) + 
                 0.019994643 * np.sin(2 * M_rad) + 
                 0.000289 * np.sin(3 * M_rad))
            
            # True longitude
            true_longitude = L + C
            true_longitude_rad = np.radians(true_longitude % 360)
            
            # Distance to Sun (AU) with eccentricity correction
            r_sun = (1.000140612 - 
                    0.016708617 * np.cos(M_rad) - 
                    0.000139589 * np.cos(2 * M_rad))
            
            # Obliquity of ecliptic (with nutation)
            obliquity = 23.439291 - 0.0130042 * T
            obliquity_rad = np.radians(obliquity)
            
            # Convert to ECI coordinates
            x_sun = r_sun * np.cos(true_longitude_rad) * self.au
            y_sun = r_sun * np.sin(true_longitude_rad) * np.cos(obliquity_rad) * self.au
            z_sun = r_sun * np.sin(true_longitude_rad) * np.sin(obliquity_rad) * self.au
            
            return np.array([x_sun, y_sun, z_sun])
            
        except Exception as e:
            self.logger.error(f"Analytical Sun position calculation failed: {e}")
            # Very basic fallback
            return np.array([self.au, 0, 0])
    
    def _compute_enhanced_eclipse_factor(self, sat_pos: np.ndarray, 
                                       sun_pos: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Compute enhanced eclipse factor with precise umbra/penumbra modeling
        
        Returns:
            Tuple of (eclipse_factor, eclipse_info_dict)
        """
        try:
            if not self.use_precise_eclipse:
                # Simple shadow test
                factor = 0.0 if self._simple_shadow_test(sat_pos, sun_pos) else 1.0
                return factor, {'type': 'umbra' if factor == 0.0 else 'sunlight'}
            
            # Precise eclipse modeling
            r_sat = np.linalg.norm(sat_pos)
            
            if r_sat <= self.earth_radius:
                # Satellite below Earth surface
                return 0.0, {'type': 'underground'}
            
            # Vector from Earth center to satellite
            sat_unit = sat_pos / r_sat
            
            # Vector from Earth center to Sun
            sun_unit = sun_pos / np.linalg.norm(sun_pos)
            
            # Angle between satellite and Sun as seen from Earth center
            cos_angle = np.dot(sat_unit, sun_unit)
            
            # If satellite is on Sun side of Earth, no eclipse
            if cos_angle > 0:
                return 1.0, {'type': 'sunlight'}
            
            # Angular radii as seen from satellite
            earth_angular_radius = np.arcsin(self.earth_radius / r_sat)
            
            # Sun angular radius as seen from satellite
            sat_to_sun = sun_pos - sat_pos
            sat_to_sun_mag = np.linalg.norm(sat_to_sun)
            sun_angular_radius = np.arctan(self.sun_radius / sat_to_sun_mag)
            
            # Vector from satellite to Earth center
            sat_to_earth = -sat_pos
            sat_to_earth_unit = sat_to_earth / np.linalg.norm(sat_to_earth)
            
            # Vector from satellite to Sun
            sat_to_sun_unit = sat_to_sun / sat_to_sun_mag
            
            # Angular separation between Earth and Sun as seen from satellite
            angular_separation = np.arccos(np.clip(
                np.dot(sat_to_earth_unit, sat_to_sun_unit), -1, 1
            ))
            
            # Eclipse classification
            if angular_separation <= (earth_angular_radius - sun_angular_radius):
                # Total eclipse (umbra)
                return 0.0, {
                    'type': 'umbra',
                    'angular_separation': np.degrees(angular_separation),
                    'earth_angular_radius': np.degrees(earth_angular_radius),
                    'sun_angular_radius': np.degrees(sun_angular_radius)
                }
            elif angular_separation >= (earth_angular_radius + sun_angular_radius):
                # No eclipse (full sunlight)
                return 1.0, {
                    'type': 'sunlight',
                    'angular_separation': np.degrees(angular_separation)
                }
            else:
                # Partial eclipse (penumbra)
                if self.use_penumbra:
                    eclipse_factor = self._compute_penumbra_factor(
                        angular_separation, earth_angular_radius, sun_angular_radius
                    )
                    return eclipse_factor, {
                        'type': 'penumbra',
                        'eclipse_factor': eclipse_factor,
                        'angular_separation': np.degrees(angular_separation)
                    }
                else:
                    # Simple binary eclipse
                    return 0.0, {'type': 'umbra_simplified'}
                    
        except Exception as e:
            self.logger.error(f"Eclipse factor computation failed: {e}")
            return 1.0, {'type': 'error', 'error': str(e)}
    
    def _compute_penumbra_factor(self, angular_separation: float,
                               earth_angular_radius: float,
                               sun_angular_radius: float) -> float:
        """Compute penumbra eclipse factor using geometric overlap"""
        try:
            # Geometric calculation of overlapping circular areas
            # This is a simplified model - full implementation would use
            # more sophisticated geometric calculations
            
            # Distance between circle centers (normalized)
            d = angular_separation
            r1 = earth_angular_radius  # Earth disk radius
            r2 = sun_angular_radius    # Sun disk radius
            
            # Check for complete overlap cases
            if d >= r1 + r2:
                return 1.0  # No overlap
            if d <= abs(r1 - r2):
                return 0.0  # Complete overlap
            
            # Partial overlap - use lens area formula
            # Area of intersection of two circles
            part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
            
            intersection_area = part1 + part2 - part3
            sun_area = np.pi * r2**2
            
            # Eclipse factor is the fraction of Sun not blocked
            eclipse_factor = 1.0 - (intersection_area / sun_area)
            
            return np.clip(eclipse_factor, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Penumbra factor calculation failed: {e}")
            # Linear approximation as fallback
            eclipse_width = 2 * sun_angular_radius
            eclipse_center = earth_angular_radius
            
            if angular_separation <= eclipse_center:
                eclipse_depth = (eclipse_center - angular_separation + sun_angular_radius) / eclipse_width
                return max(0.0, 1.0 - eclipse_depth)
            else:
                eclipse_depth = (angular_separation - eclipse_center + sun_angular_radius) / eclipse_width
                return max(0.0, eclipse_depth)
    
    def _simple_shadow_test(self, sat_pos: np.ndarray, sun_pos: np.ndarray) -> bool:
        """Simple shadow test for fallback"""
        try:
            r_sat = np.linalg.norm(sat_pos)
            sun_dir = sun_pos / np.linalg.norm(sun_pos)
            
            # Project satellite position onto Sun direction
            sat_proj = np.dot(sat_pos, sun_dir)
            
            # If satellite is on Sun side of Earth, not in shadow
            if sat_proj > 0:
                return False
            
            # Distance from satellite to Sun-Earth line
            perp_dist = np.linalg.norm(sat_pos - sat_proj * sun_dir)
            
            # Earth's shadow radius at satellite distance
            shadow_radius = self.earth_radius
            
            return perp_dist < shadow_radius
            
        except Exception as e:
            self.logger.error(f"Simple shadow test failed: {e}")
            return False
    
    def _compute_seasonal_solar_flux(self, datetime_utc: datetime, 
                                   sun_distance: float) -> float:
        """Compute solar flux with seasonal Earth-Sun distance variations"""
        try:
            # Day of year for seasonal calculation
            day_of_year = datetime_utc.timetuple().tm_yday
            
            # Earth's orbital position (approximate)
            # Perihelion around January 3 (day 3)
            orbital_angle = 2 * np.pi * (day_of_year - 3) / 365.25
            
            # Earth-Sun distance variation due to orbital eccentricity
            distance_factor = (1 - self.earth_orbit_eccentricity * np.cos(orbital_angle))
            
            # Solar flux varies as 1/r²
            flux_variation = 1.0 / (distance_factor**2)
            
            # Base solar flux at current distance
            distance_au = sun_distance / self.au
            base_flux = self.solar_flux_1au / (distance_au**2)
            
            # Apply seasonal variation
            seasonal_flux = base_flux * flux_variation
            
            return seasonal_flux
            
        except Exception as e:
            self.logger.warning(f"Seasonal flux calculation failed: {e}")
            # Fallback to simple inverse square law
            distance_au = sun_distance / self.au
            return self.solar_flux_1au / (distance_au**2)
    
    def _compute_adaptive_srp_coefficient(self, satellite_props: SatelliteProperties,
                                        position_eci: np.ndarray,
                                        sun_direction: np.ndarray) -> float:
        """Compute adaptive SRP coefficient based on satellite characteristics"""
        try:
            if not self.use_adaptive_coefficients:
                return satellite_props.srp_coefficient
            
            base_cr = satellite_props.srp_coefficient
            
            # Satellite type adjustments
            if satellite_props.satellite_type == SatelliteType.SPACE_STATION:
                # Large satellites with complex geometry
                # Solar panels and complex surfaces
                cr_adjustment = 1.1  # 10% increase for complex geometry
                
            elif satellite_props.satellite_type == SatelliteType.EARTH_OBSERVATION:
                # Often have large solar panels
                cr_adjustment = 1.05  # 5% increase
                
            elif satellite_props.satellite_type == SatelliteType.COMMUNICATION:
                # Typically have solar panels and antennas
                cr_adjustment = 1.02  # 2% increase
                
            elif satellite_props.satellite_type == SatelliteType.SCIENTIFIC:
                # Often compact with fewer appendages
                cr_adjustment = 0.98  # 2% decrease
                
            else:
                cr_adjustment = 1.0  # No adjustment for unknown types
            
            # Orbital regime adjustments
            if satellite_props.orbital_regime == OrbitalRegime.VERY_LOW_LEO:
                # Atmospheric effects may reduce effective SRP
                cr_adjustment *= 0.95
                
            elif satellite_props.orbital_regime == OrbitalRegime.HIGH_LEO:
                # Less atmospheric interference
                cr_adjustment *= 1.02
            
            # Confidence-based uncertainty
            confidence = satellite_props.characterization_confidence
            if confidence < 0.5:
                # Add conservative margin for poorly characterized satellites
                cr_adjustment *= 1.1
            
            adaptive_cr = base_cr * cr_adjustment
            
            # Ensure reasonable bounds
            adaptive_cr = np.clip(adaptive_cr, 0.5, 2.5)
            
            return adaptive_cr
            
        except Exception as e:
            self.logger.warning(f"Adaptive SRP coefficient calculation failed: {e}")
            return satellite_props.srp_coefficient
    
    def _apply_attitude_corrections(self, srp_accel_basic: np.ndarray,
                                  satellite_props: SatelliteProperties,
                                  sun_direction: np.ndarray,
                                  velocity_eci: np.ndarray) -> np.ndarray:
        """Apply attitude-dependent SRP corrections"""
        try:
            if self.attitude_modeling == 'none':
                return srp_accel_basic
            
            # For unknown attitude, use conservative modeling
            if self.attitude_modeling == 'conservative':
                # Assume random tumbling - reduce SRP by average factor
                attitude_factor = 0.7  # Conservative 30% reduction
                
            elif self.attitude_modeling == 'nadir_pointing':
                # Nadir-pointing satellites (common for Earth observation)
                # SRP varies with orbital position relative to Sun
                attitude_factor = 0.8  # Moderate reduction
                
            elif self.attitude_modeling == 'sun_pointing':
                # Sun-pointing satellites (solar panels always face Sun)
                attitude_factor = 1.2  # Increased SRP
                
            elif self.attitude_modeling == 'inertial':
                # Inertially fixed attitude
                attitude_factor = 0.9  # Slight reduction
                
            else:
                # Default conservative approach
                attitude_factor = 0.8
            
            # Apply satellite type specific attitude corrections
            if satellite_props.satellite_type == SatelliteType.SPACE_STATION:
                # Large stations often have attitude control
                attitude_factor *= 1.1
                
            elif satellite_props.satellite_type == SatelliteType.COMMUNICATION:
                # Often have Earth-pointing attitude
                attitude_factor *= 0.9
            
            corrected_accel = srp_accel_basic * attitude_factor
            
            return corrected_accel
            
        except Exception as e:
            self.logger.warning(f"Attitude correction failed: {e}")
            return srp_accel_basic
    
    def get_srp_diagnostics(self, satellite_props: SatelliteProperties,
                          position_eci: np.ndarray, datetime_utc: datetime) -> Dict[str, Any]:
        """Get comprehensive SRP diagnostics"""
        try:
            # Compute SRP acceleration and get diagnostics
            srp_accel, diagnostics = self.compute_srp_acceleration(
                satellite_props, position_eci, np.zeros(3), datetime_utc
            )
            
            # Add additional diagnostic information
            sun_pos = self._get_precise_sun_position(datetime_utc)
            sat_to_sun = sun_pos - position_eci
            
            diagnostics.update({
                'srp_acceleration_magnitude': np.linalg.norm(srp_accel),
                'sun_satellite_distance_km': np.linalg.norm(sat_to_sun) / 1000,
                'satellite_type': satellite_props.satellite_type.value,
                'orbital_regime': satellite_props.orbital_regime.value,
                'srp_area_m2': satellite_props.srp_area,
                'satellite_mass_kg': satellite_props.mass,
                'base_srp_coefficient': satellite_props.srp_coefficient,
                'model_configuration': {
                    'precise_eclipse': self.use_precise_eclipse,
                    'penumbra_modeling': self.use_penumbra,
                    'adaptive_coefficients': self.use_adaptive_coefficients,
                    'attitude_modeling': self.attitude_modeling
                }
            })
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"SRP diagnostics failed: {e}")
            return {'error': str(e)}
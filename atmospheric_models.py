import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import logging
from space_weather import SpaceWeatherData
from satellite_characterizer import SatelliteProperties, SatelliteType, OrbitalRegime

class EnhancedAdaptiveAtmosphericDragModel:
    """
    Adaptive atmospheric drag model that integrates real-time space weather data
    and adjusts drag coefficients based on satellite characteristics for satellite-agnostic operation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive atmospheric drag model"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize space weather data provider
        self.space_weather = SpaceWeatherData()
        
        # Model constants
        self.R_earth = 6371.0  # km
        self.scale_height_base = 8.5  # km
        
        # Adaptive parameters
        self.auto_update_space_weather = self.config.get('auto_update_space_weather', True)
        self.space_weather_update_interval = self.config.get('space_weather_update_interval', 3600)  # seconds
        self.last_space_weather_update = None
        
        # Density model selection
        self.primary_model = NRLMSISE00()
        self.fallback_model = ExponentialAtmosphere()
        
        # Drag coefficient adaptation parameters
        self.drag_coeff_adaptation = self.config.get('drag_coeff_adaptation', True)
        self.uncertainty_propagation = self.config.get('uncertainty_propagation', True)
        
        self.logger.info("Adaptive atmospheric drag model initialized")
    
    def get_drag_acceleration(self, position: np.ndarray, velocity: np.ndarray, 
                            satellite_props: SatelliteProperties, 
                            datetime_utc: datetime) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calculate drag acceleration with adaptive modeling based on satellite characteristics
        
        Args:
            position: Position vector in ECEF (m)
            velocity: Velocity vector in ECEF (m/s)
            satellite_props: Satellite properties from characterization
            datetime_utc: Current UTC datetime
            
        Returns:
            Tuple of (drag_acceleration_vector, uncertainty_metrics)
        """
        try:
            # Update space weather data if needed
            if self.auto_update_space_weather:
                self._update_space_weather_if_needed()
            
            # Calculate atmospheric density with space weather integration
            density, density_uncertainty = self._get_adaptive_density(
                position, datetime_utc, satellite_props
            )
            
            # Get adaptive drag coefficient
            drag_coeff, drag_coeff_uncertainty = self._get_adaptive_drag_coefficient(
                satellite_props, position, velocity
            )
            
            # Calculate relative velocity (accounting for atmospheric rotation)
            v_rel = self._calculate_relative_velocity(position, velocity)
            v_rel_mag = np.linalg.norm(v_rel)
            
            if v_rel_mag < 1e-6:  # Avoid division by zero
                return np.zeros(3), {'density_uncertainty': 0.0, 'drag_coeff_uncertainty': 0.0}
            
            # Drag acceleration calculation
            # F_drag = -0.5 * rho * Cd * A * v_rel * |v_rel|
            # a_drag = F_drag / m
            drag_area = satellite_props.drag_area
            mass = satellite_props.mass
            
            drag_force_magnitude = 0.5 * density * drag_coeff * drag_area * v_rel_mag**2
            drag_acceleration_magnitude = drag_force_magnitude / mass
            
            # Direction opposite to relative velocity
            drag_direction = -v_rel / v_rel_mag
            drag_acceleration = drag_acceleration_magnitude * drag_direction
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_drag_uncertainty(
                density_uncertainty, drag_coeff_uncertainty, satellite_props
            )
            
            self.logger.debug(f"Drag acceleration calculated: magnitude={drag_acceleration_magnitude:.2e} m/s²")
            
            return drag_acceleration, uncertainty_metrics
            
        except Exception as e:
            self.logger.error(f"Drag acceleration calculation failed: {e}")
            return np.zeros(3), {'density_uncertainty': 1.0, 'drag_coeff_uncertainty': 1.0}
    
    def _update_space_weather_if_needed(self):
        """Update space weather data if needed"""
        try:
            current_time = datetime.utcnow()
            
            if (self.last_space_weather_update is None or 
                (current_time - self.last_space_weather_update).total_seconds() > self.space_weather_update_interval):
                
                if self.space_weather.update():
                    self.last_space_weather_update = current_time
                    self.logger.info("Space weather data updated successfully")
                else:
                    self.logger.warning("Space weather update failed, using cached data")
                    
        except Exception as e:
            self.logger.warning(f"Space weather update error: {e}")
    
    def _get_adaptive_density(self, position: np.ndarray, datetime_utc: datetime, 
                            satellite_props: SatelliteProperties) -> Tuple[float, float]:
        """
        Get atmospheric density with adaptive modeling based on satellite characteristics
        
        Returns:
            Tuple of (density, density_uncertainty)
        """
        try:
            # Convert position to geodetic coordinates
            altitude_km, latitude, longitude = self._ecef_to_geodetic(position)
            
            # Get current space weather data
            sw_data = self.space_weather.get_current_data()
            f107 = sw_data['f107']
            kp = sw_data['kp']
            
            # Use primary model (NRLMSISE-00) with space weather data
            try:
                density = self.primary_model.get_density(
                    altitude_km, latitude, longitude, datetime_utc, f107, kp
                )
                
                # Calculate density uncertainty based on space weather data quality
                density_uncertainty = self._calculate_density_uncertainty(
                    sw_data, satellite_props, altitude_km
                )
                
            except Exception as e:
                self.logger.warning(f"Primary density model failed: {e}, using fallback")
                density = self.fallback_model.get_density(altitude_km)
                density_uncertainty = 0.5  # High uncertainty for fallback
            
            # Apply altitude-dependent scaling based on satellite characteristics
            density_scaling, scaling_uncertainty = self._get_altitude_dependent_scaling(
                altitude_km, satellite_props
            )
            
            density *= density_scaling
            density_uncertainty = np.sqrt(density_uncertainty**2 + scaling_uncertainty**2)
            
            return density, density_uncertainty
            
        except Exception as e:
            self.logger.error(f"Adaptive density calculation failed: {e}")
            return 1e-12, 1.0  # Very low density with high uncertainty
    
    def _get_adaptive_drag_coefficient(self, satellite_props: SatelliteProperties, 
                                     position: np.ndarray, velocity: np.ndarray) -> Tuple[float, float]:
        """
        Get adaptive drag coefficient based on satellite characteristics and flight conditions
        
        Returns:
            Tuple of (drag_coefficient, uncertainty)
        """
        try:
            if not self.drag_coeff_adaptation:
                return satellite_props.drag_coefficient, 0.1
            
            # Base drag coefficient from satellite characterization
            base_cd = satellite_props.drag_coefficient
            
            # Adaptive adjustments based on satellite type
            type_adjustment, type_uncertainty = self._get_satellite_type_adjustment(satellite_props)
            
            # Altitude-dependent adjustments
            altitude_km, _, _ = self._ecef_to_geodetic(position)
            altitude_adjustment, altitude_uncertainty = self._get_altitude_drag_adjustment(altitude_km)
            
            # Velocity-dependent adjustments (Mach number effects)
            velocity_adjustment, velocity_uncertainty = self._get_velocity_drag_adjustment(
                position, velocity
            )
            
            # Combine adjustments
            total_adjustment = type_adjustment * altitude_adjustment * velocity_adjustment
            adapted_cd = base_cd * total_adjustment
            
            # Combine uncertainties
            total_uncertainty = np.sqrt(
                type_uncertainty**2 + 
                altitude_uncertainty**2 + 
                velocity_uncertainty**2 +
                (satellite_props.parameter_confidence.get('drag_coefficient', 0.2) * 0.5)**2
            )
            
            # Apply physical bounds
            adapted_cd = np.clip(adapted_cd, 1.0, 3.5)
            
            return adapted_cd, total_uncertainty
            
        except Exception as e:
            self.logger.warning(f"Adaptive drag coefficient calculation failed: {e}")
            return satellite_props.drag_coefficient, 0.3
    
    def _get_satellite_type_adjustment(self, satellite_props: SatelliteProperties) -> Tuple[float, float]:
        """Get drag coefficient adjustment based on satellite type"""
        try:
            # Satellite type specific adjustments based on typical shapes and orientations
            type_adjustments = {
                SatelliteType.SPACE_STATION: (1.1, 0.15),  # Complex geometry, higher drag
                SatelliteType.EARTH_OBSERVATION: (0.95, 0.10),  # Streamlined, lower drag
                SatelliteType.COMMUNICATION: (1.0, 0.12),  # Moderate complexity
                SatelliteType.SCIENTIFIC: (0.9, 0.20),  # Often small and streamlined
                SatelliteType.UNKNOWN: (1.0, 0.25)  # High uncertainty
            }
            
            adjustment, uncertainty = type_adjustments.get(
                satellite_props.satellite_type, (1.0, 0.25)
            )
            
            # Adjust uncertainty based on characterization confidence
            confidence_factor = satellite_props.characterization_confidence
            uncertainty *= (1.0 - confidence_factor * 0.5)
            
            return adjustment, uncertainty
            
        except Exception as e:
            self.logger.warning(f"Satellite type adjustment failed: {e}")
            return 1.0, 0.25
    
    def _get_altitude_drag_adjustment(self, altitude_km: float) -> Tuple[float, float]:
        """Get drag coefficient adjustment based on altitude (atmospheric composition effects)"""
        try:
            # Altitude-dependent drag coefficient adjustments
            # Based on atmospheric composition changes with altitude
            
            if altitude_km < 200:
                # Dense atmosphere, molecular effects
                adjustment = 1.05
                uncertainty = 0.08
            elif altitude_km < 300:
                # Transition region
                adjustment = 1.02
                uncertainty = 0.06
            elif altitude_km < 500:
                # Typical LEO region
                adjustment = 1.0
                uncertainty = 0.05
            elif altitude_km < 700:
                # Higher LEO, more atomic oxygen
                adjustment = 0.98
                uncertainty = 0.07
            else:
                # Very high LEO
                adjustment = 0.95
                uncertainty = 0.10
            
            return adjustment, uncertainty
            
        except Exception as e:
            self.logger.warning(f"Altitude drag adjustment failed: {e}")
            return 1.0, 0.10
    
    def _get_velocity_drag_adjustment(self, position: np.ndarray, velocity: np.ndarray) -> Tuple[float, float]:
        """Get drag coefficient adjustment based on velocity (Mach number effects)"""
        try:
            # Calculate relative velocity magnitude
            v_rel = self._calculate_relative_velocity(position, velocity)
            v_rel_mag = np.linalg.norm(v_rel)
            
            # Estimate local speed of sound (very approximate for upper atmosphere)
            altitude_km, _, _ = self._ecef_to_geodetic(position)
            
            # Temperature estimate for speed of sound
            if altitude_km < 200:
                temp = 200  # K
            elif altitude_km < 500:
                temp = 800 + (altitude_km - 200) * 2  # K, rough estimate
            else:
                temp = 1400  # K, exospheric temperature
            
            # Speed of sound in air (approximate)
            gamma = 1.4  # Heat capacity ratio
            R_specific = 287  # J/(kg·K) for air
            speed_of_sound = np.sqrt(gamma * R_specific * temp)
            
            # Mach number
            mach_number = v_rel_mag / speed_of_sound
            
            # Drag coefficient adjustment based on Mach number
            # For hypersonic flow (typical for satellites)
            if mach_number > 5:
                adjustment = 1.0 + 0.02 * np.log(mach_number / 5)  # Slight increase at very high Mach
                uncertainty = 0.05
            else:
                adjustment = 1.0
                uncertainty = 0.03
            
            return adjustment, uncertainty
            
        except Exception as e:
            self.logger.warning(f"Velocity drag adjustment failed: {e}")
            return 1.0, 0.05
    
    def _calculate_relative_velocity(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Calculate velocity relative to rotating atmosphere"""
        try:
            # Earth rotation rate
            omega_earth = 7.2921159e-5  # rad/s
            
            # Atmospheric rotation velocity at satellite position
            # v_atm = omega × r
            omega_vector = np.array([0, 0, omega_earth])
            v_atmosphere = np.cross(omega_vector, position)
            
            # Relative velocity
            v_relative = velocity - v_atmosphere
            
            return v_relative
            
        except Exception as e:
            self.logger.warning(f"Relative velocity calculation failed: {e}")
            return velocity  # Fallback to inertial velocity
    
    def _ecef_to_geodetic(self, position: np.ndarray) -> Tuple[float, float, float]:
        """Convert ECEF position to geodetic coordinates (simplified)"""
        try:
            x, y, z = position
            
            # Longitude
            longitude = np.degrees(np.arctan2(y, x))
            
            # Latitude (simplified, assuming spherical Earth)
            r = np.linalg.norm(position)
            latitude = np.degrees(np.arcsin(z / r))
            
            # Altitude
            altitude_m = r - self.R_earth * 1000  # Convert to meters
            altitude_km = altitude_m / 1000
            
            return altitude_km, latitude, longitude
            
        except Exception as e:
            self.logger.warning(f"Coordinate conversion failed: {e}")
            return 400.0, 0.0, 0.0  # Default values
    
    def _get_altitude_dependent_scaling(self, altitude_km: float, 
                                      satellite_props: SatelliteProperties) -> Tuple[float, float]:
        """Get altitude-dependent density scaling based on satellite characteristics"""
        try:
            # Orbital regime specific scaling
            regime_scalings = {
                OrbitalRegime.VERY_LOW_LEO: (1.05, 0.08),  # Higher density region
                OrbitalRegime.LOW_LEO: (1.0, 0.05),        # Reference region
                OrbitalRegime.MID_LEO: (0.98, 0.06),       # Slightly lower density
                OrbitalRegime.HIGH_LEO: (0.95, 0.10)       # Lower density, higher uncertainty
            }
            
            scaling, uncertainty = regime_scalings.get(
                satellite_props.orbital_regime, (1.0, 0.08)
            )
            
            # Additional scaling based on satellite type (different atmospheric interaction)
            if satellite_props.satellite_type == SatelliteType.SPACE_STATION:
                # Large satellites may experience different atmospheric conditions
                scaling *= 1.02
                uncertainty += 0.03
            elif satellite_props.satellite_type == SatelliteType.SCIENTIFIC:
                # Small satellites may be more sensitive to local variations
                uncertainty += 0.05
            
            return scaling, uncertainty
            
        except Exception as e:
            self.logger.warning(f"Altitude scaling calculation failed: {e}")
            return 1.0, 0.10
    
    def _calculate_density_uncertainty(self, sw_data: Dict[str, Any], 
                                     satellite_props: SatelliteProperties, 
                                     altitude_km: float) -> float:
        """Calculate density uncertainty based on space weather data quality and satellite characteristics"""
        try:
            # Base uncertainty from space weather data age
            data_age_hours = sw_data.get('data_age_hours', 999)
            
            if data_age_hours < 6:
                age_uncertainty = 0.05
            elif data_age_hours < 24:
                age_uncertainty = 0.10
            elif data_age_hours < 72:
                age_uncertainty = 0.20
            else:
                age_uncertainty = 0.40
            
            # Uncertainty from space weather activity level
            kp = sw_data.get('kp', 3.0)
            if kp < 3:
                activity_uncertainty = 0.05
            elif kp < 5:
                activity_uncertainty = 0.10
            elif kp < 7:
                activity_uncertainty = 0.20
            else:
                activity_uncertainty = 0.35
            
            # Altitude-dependent uncertainty
            if altitude_km < 300:
                altitude_uncertainty = 0.08
            elif altitude_km < 500:
                altitude_uncertainty = 0.12
            else:
                altitude_uncertainty = 0.18
            
            # Satellite characterization uncertainty contribution
            char_uncertainty = (1.0 - satellite_props.characterization_confidence) * 0.15
            
            # Combine uncertainties
            total_uncertainty = np.sqrt(
                age_uncertainty**2 + 
                activity_uncertainty**2 + 
                altitude_uncertainty**2 + 
                char_uncertainty**2
            )
            
            return min(total_uncertainty, 0.8)  # Cap at 80% uncertainty
            
        except Exception as e:
            self.logger.warning(f"Density uncertainty calculation failed: {e}")
            return 0.3  # Default moderate uncertainty
    
    def _calculate_drag_uncertainty(self, density_uncertainty: float, 
                                  drag_coeff_uncertainty: float, 
                                  satellite_props: SatelliteProperties) -> Dict[str, float]:
        """Calculate overall drag acceleration uncertainty metrics"""
        try:
            # Area uncertainty from satellite characterization
            area_uncertainty = satellite_props.uncertainty_bounds.get('drag_area', (0, 0))
            if isinstance(area_uncertainty, tuple) and len(area_uncertainty) == 2:
                area_relative_uncertainty = (area_uncertainty[1] - area_uncertainty[0]) / (2 * satellite_props.drag_area)
            else:
                area_relative_uncertainty = 0.2
            
            # Mass uncertainty
            mass_uncertainty = satellite_props.uncertainty_bounds.get('mass', (0, 0))
            if isinstance(mass_uncertainty, tuple) and len(mass_uncertainty) == 2:
                mass_relative_uncertainty = (mass_uncertainty[1] - mass_uncertainty[0]) / (2 * satellite_props.mass)
            else:
                mass_relative_uncertainty = 0.15
            
            # Total drag acceleration uncertainty
            # a_drag ∝ (rho * Cd * A) / m
            total_relative_uncertainty = np.sqrt(
                density_uncertainty**2 + 
                drag_coeff_uncertainty**2 + 
                area_relative_uncertainty**2 + 
                mass_relative_uncertainty**2
            )
            
            return {
                'density_uncertainty': density_uncertainty,
                'drag_coeff_uncertainty': drag_coeff_uncertainty,
                'area_uncertainty': area_relative_uncertainty,
                'mass_uncertainty': mass_relative_uncertainty,
                'total_drag_uncertainty': total_relative_uncertainty
            }
            
        except Exception as e:
            self.logger.warning(f"Drag uncertainty calculation failed: {e}")
            return {
                'density_uncertainty': density_uncertainty,
                'drag_coeff_uncertainty': drag_coeff_uncertainty,
                'total_drag_uncertainty': 0.5
            }

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
        Get atmospheric density at specified conditions with enhanced space weather integration
        
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
            
            # Enhanced NRLMSISE-00 implementation with improved space weather integration
            # Base exponential atmosphere with altitude-dependent scale heights
            base_density = self._enhanced_exponential_atmosphere(altitude_km)
            
            # Enhanced space weather corrections
            solar_correction = self._enhanced_solar_flux_correction(f107, altitude_km, datetime_utc)
            geomagnetic_correction = self._enhanced_geomagnetic_correction(kp, altitude_km, latitude)
            
            # Improved diurnal and seasonal variations
            diurnal_correction = self._enhanced_diurnal_variation(
                longitude, datetime_utc, altitude_km, latitude
            )
            seasonal_correction = self._enhanced_seasonal_variation(
                latitude, datetime_utc, altitude_km
            )
            
            # Semi-annual variation (important for LEO satellites)
            semiannual_correction = self._semiannual_variation(datetime_utc, altitude_km)
            
            # Combine all corrections with improved weighting
            density = (base_density * 
                      solar_correction * 
                      geomagnetic_correction * 
                      diurnal_correction * 
                      seasonal_correction *
                      semiannual_correction)
            
            return max(density, 1e-15)  # Minimum density threshold
            
        except Exception as e:
            self.logger.error(f"Density calculation error: {e}")
            return self._fallback_density(altitude_km)
    
    def _enhanced_exponential_atmosphere(self, altitude_km: float) -> float:
        """Enhanced exponential atmosphere model with improved altitude scaling"""
        if altitude_km < 0:
            altitude_km = 0
        
        # Enhanced altitude-dependent density model
        if altitude_km < 120:
            # Lower atmosphere - use standard atmosphere
            rho_0 = 1.225  # kg/m^3 at sea level
            H = 8.5  # km
            density = rho_0 * np.exp(-altitude_km / H)
        elif altitude_km < 200:
            # Transition region - improved modeling
            rho_120 = 5.6e-6  # kg/m^3 at 120 km
            H = 42.0  # km, larger scale height
            density = rho_120 * np.exp(-(altitude_km - 120) / H)
        elif altitude_km < 300:
            # Thermosphere lower region
            rho_200 = 2.5e-10  # kg/m^3 at 200 km
            H = 60.0  # km
            density = rho_200 * np.exp(-(altitude_km - 200) / H)
        elif altitude_km < 500:
            # Thermosphere middle region
            rho_300 = 1.9e-11  # kg/m^3 at 300 km
            H = 100.0  # km
            density = rho_300 * np.exp(-(altitude_km - 300) / H)
        elif altitude_km < 700:
            # Thermosphere upper region
            rho_500 = 6.0e-13  # kg/m^3 at 500 km
            H = 150.0  # km
            density = rho_500 * np.exp(-(altitude_km - 500) / H)
        else:
            # Exosphere transition
            rho_700 = 1.4e-14  # kg/m^3 at 700 km
            H = 200.0  # km
            density = rho_700 * np.exp(-(altitude_km - 700) / H)
        
        return density
    
    def _exponential_atmosphere(self, altitude_km: float) -> float:
        """Base exponential atmosphere model (legacy method)"""
        return self._enhanced_exponential_atmosphere(altitude_km)
    
    def _enhanced_solar_flux_correction(self, f107: float, altitude_km: float, datetime_utc: datetime) -> float:
        """Enhanced solar flux correction with improved altitude and temporal dependencies"""
        # Normalized solar flux (relative to quiet conditions)
        f107_norm = f107 / 150.0
        
        # Enhanced altitude-dependent solar influence
        if altitude_km < 150:
            solar_influence = 0.05  # Minimal solar influence in lower thermosphere
        elif altitude_km < 200:
            # Smooth transition
            solar_influence = 0.05 + 0.15 * (altitude_km - 150) / 50
        elif altitude_km < 300:
            # Strong solar influence region
            solar_influence = 0.2 + 0.4 * (altitude_km - 200) / 100
        elif altitude_km < 500:
            # Peak solar influence
            solar_influence = 0.6 + 0.3 * (altitude_km - 300) / 200
        elif altitude_km < 700:
            # Gradual decrease
            solar_influence = 0.9 - 0.2 * (altitude_km - 500) / 200
        else:
            # High altitude, moderate influence
            solar_influence = 0.7
        
        # Solar cycle phase adjustment (11-year cycle)
        year = datetime_utc.year
        solar_cycle_phase = (year - 2008) % 11  # 2008 was near solar minimum
        
        # Adjust influence based on solar cycle phase
        if solar_cycle_phase < 4:  # Solar minimum to rising
            cycle_factor = 0.8 + 0.2 * solar_cycle_phase / 4
        elif solar_cycle_phase < 7:  # Solar maximum
            cycle_factor = 1.0 + 0.3 * np.sin(np.pi * (solar_cycle_phase - 4) / 3)
        else:  # Solar maximum to minimum
            cycle_factor = 1.3 - 0.5 * (solar_cycle_phase - 7) / 4
        
        solar_influence *= cycle_factor
        
        # Enhanced correction factor with saturation effects
        if f107_norm > 2.0:  # Very high solar activity
            correction = 1.0 + solar_influence * (1.5 + 0.5 * np.log(f107_norm / 2.0))
        else:
            correction = 1.0 + solar_influence * (f107_norm - 1.0)
        
        return max(correction, 0.1)
    
    def _solar_flux_correction(self, f107: float, altitude_km: float) -> float:
        """Solar flux correction factor (legacy method)"""
        return self._enhanced_solar_flux_correction(f107, altitude_km, datetime.utcnow())
    
    def _enhanced_geomagnetic_correction(self, kp: float, altitude_km: float, latitude: float) -> float:
        """Enhanced geomagnetic activity correction with latitude dependence"""
        # Normalized Kp index with improved scaling
        kp_norm = kp / 4.0  # Use Kp=4 as reference (moderate activity)
        
        # Enhanced altitude-dependent geomagnetic influence
        if altitude_km < 200:
            geo_influence = 0.02  # Minimal influence at low altitudes
        elif altitude_km < 300:
            # Gradual increase
            geo_influence = 0.02 + 0.08 * (altitude_km - 200) / 100
        elif altitude_km < 400:
            # Strong influence region
            geo_influence = 0.1 + 0.15 * (altitude_km - 300) / 100
        elif altitude_km < 600:
            # Peak influence
            geo_influence = 0.25 + 0.2 * (altitude_km - 400) / 200
        else:
            # High altitude, strong influence
            geo_influence = 0.45
        
        # Latitude-dependent geomagnetic effects
        # Higher effects at high latitudes (auroral zones)
        abs_latitude = abs(latitude)
        if abs_latitude > 60:  # Polar regions
            latitude_factor = 1.5 + 0.5 * (abs_latitude - 60) / 30
        elif abs_latitude > 45:  # Sub-polar regions
            latitude_factor = 1.0 + 0.5 * (abs_latitude - 45) / 15
        else:  # Equatorial and mid-latitude regions
            latitude_factor = 1.0
        
        geo_influence *= latitude_factor
        
        # Enhanced correction with storm effects
        if kp > 6:  # Geomagnetic storm conditions
            storm_enhancement = 1.0 + 0.3 * (kp - 6) / 3
            geo_influence *= storm_enhancement
        
        # Correction factor (geomagnetic heating increases density)
        correction = 1.0 + geo_influence * (kp_norm - 1.0)
        
        return max(correction, 0.3)
    
    def _geomagnetic_correction(self, kp: float, altitude_km: float) -> float:
        """Geomagnetic activity correction factor (legacy method)"""
        return self._enhanced_geomagnetic_correction(kp, altitude_km, 0.0)
    
    def _enhanced_diurnal_variation(self, longitude: float, datetime_utc: datetime, 
                                  altitude_km: float, latitude: float) -> float:
        """Enhanced diurnal (daily) density variation with latitude dependence"""
        # Local solar time calculation with improved precision
        hour_utc = datetime_utc.hour + datetime_utc.minute / 60.0 + datetime_utc.second / 3600.0
        hour_local = hour_utc + longitude / 15.0
        hour_local = hour_local % 24
        
        # Enhanced altitude-dependent diurnal amplitude
        if altitude_km < 150:
            amplitude = 0.05  # Minimal diurnal variation
        elif altitude_km < 200:
            amplitude = 0.05 + 0.1 * (altitude_km - 150) / 50
        elif altitude_km < 300:
            amplitude = 0.15 + 0.2 * (altitude_km - 200) / 100
        elif altitude_km < 500:
            amplitude = 0.35 + 0.25 * (altitude_km - 300) / 200
        elif altitude_km < 700:
            amplitude = 0.6 - 0.1 * (altitude_km - 500) / 200
        else:
            amplitude = 0.5
        
        # Latitude-dependent amplitude (stronger at low latitudes)
        abs_latitude = abs(latitude)
        if abs_latitude < 30:  # Equatorial region
            latitude_factor = 1.0 + 0.3 * (30 - abs_latitude) / 30
        elif abs_latitude < 60:  # Mid-latitudes
            latitude_factor = 1.0
        else:  # Polar regions
            latitude_factor = 0.7 - 0.2 * (abs_latitude - 60) / 30
        
        amplitude *= latitude_factor
        
        # Enhanced diurnal model with asymmetry
        # Peak density around 14:00-15:00 local time with asymmetric shape
        phase_primary = 2 * np.pi * (hour_local - 14.5) / 24.0
        phase_secondary = 2 * np.pi * (hour_local - 2.0) / 24.0  # Secondary minimum
        
        # Primary diurnal component
        primary_component = amplitude * np.cos(phase_primary)
        
        # Secondary component (weaker, represents pre-dawn minimum)
        secondary_component = -0.2 * amplitude * np.cos(phase_secondary)
        
        correction = 1.0 + primary_component + secondary_component
        
        return max(correction, 0.3)  # Prevent negative densities
    
    def _diurnal_variation(self, longitude: float, datetime_utc: datetime, 
                          altitude_km: float) -> float:
        """Diurnal (daily) density variation (legacy method)"""
        return self._enhanced_diurnal_variation(longitude, datetime_utc, altitude_km, 0.0)
    
    def _enhanced_seasonal_variation(self, latitude: float, datetime_utc: datetime, 
                                   altitude_km: float) -> float:
        """Enhanced seasonal density variation with improved modeling"""
        # Day of year with leap year handling
        day_of_year = datetime_utc.timetuple().tm_yday
        year = datetime_utc.year
        days_in_year = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
        
        # Enhanced latitude-dependent seasonal effects
        lat_rad = np.radians(latitude)
        
        # Seasonal amplitude depends on latitude and altitude
        if altitude_km < 200:
            base_amplitude = 0.08
        elif altitude_km < 300:
            base_amplitude = 0.08 + 0.12 * (altitude_km - 200) / 100
        elif altitude_km < 500:
            base_amplitude = 0.2 + 0.15 * (altitude_km - 300) / 200
        else:
            base_amplitude = 0.35
        
        # Latitude dependence with hemisphere asymmetry
        if latitude >= 0:  # Northern hemisphere
            # Peak density in northern winter (around day 15, January 15)
            peak_day = 15
            lat_factor = abs(np.sin(lat_rad)) * (1.0 + 0.2 * np.cos(lat_rad))
        else:  # Southern hemisphere
            # Peak density in southern winter (around day 195, July 15)
            peak_day = 195
            lat_factor = abs(np.sin(lat_rad)) * (1.0 + 0.2 * np.cos(lat_rad))
        
        amplitude = base_amplitude * lat_factor
        
        # Enhanced seasonal model with multiple harmonics
        # Primary annual component
        phase_annual = 2 * np.pi * (day_of_year - peak_day) / days_in_year
        annual_component = amplitude * np.cos(phase_annual)
        
        # Semi-annual component (global effect)
        phase_semiannual = 4 * np.pi * (day_of_year - 80) / days_in_year  # Peak around day 80 and 263
        semiannual_amplitude = 0.3 * amplitude
        semiannual_component = semiannual_amplitude * np.cos(phase_semiannual)
        
        correction = 1.0 + annual_component + semiannual_component
        
        return max(correction, 0.5)
    
    def _semiannual_variation(self, datetime_utc: datetime, altitude_km: float) -> float:
        """Semi-annual density variation (important for LEO satellites)"""
        try:
            day_of_year = datetime_utc.timetuple().tm_yday
            year = datetime_utc.year
            days_in_year = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
            
            # Semi-annual variation amplitude depends on altitude
            if altitude_km < 200:
                amplitude = 0.03
            elif altitude_km < 400:
                amplitude = 0.03 + 0.07 * (altitude_km - 200) / 200
            elif altitude_km < 600:
                amplitude = 0.1 + 0.05 * (altitude_km - 400) / 200
            else:
                amplitude = 0.15
            
            # Semi-annual phase (peaks around equinoxes: day 80 and day 263)
            phase = 4 * np.pi * (day_of_year - 80) / days_in_year
            correction = 1.0 + amplitude * np.cos(phase)
            
            return correction
            
        except Exception as e:
            self.logger.warning(f"Semi-annual variation calculation failed: {e}")
            return 1.0
    
    def _seasonal_variation(self, latitude: float, datetime_utc: datetime, 
                           altitude_km: float) -> float:
        """Seasonal density variation (legacy method)"""
        return self._enhanced_seasonal_variation(latitude, datetime_utc, altitude_km)
    
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

import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging

from atmospheric_models import NRLMSISE00, ExponentialAtmosphere
from space_weather import SpaceWeatherData
from satellite_characterizer import SatelliteProperties, OrbitalRegime

class AdaptiveAtmosphericModel:
    """
    Adaptive atmospheric drag model that adjusts density calculations
    based on satellite characteristics, orbital regime, and real-time space weather
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive atmospheric model"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize atmospheric models
        self.nrlmsise = NRLMSISE00()
        self.exponential = ExponentialAtmosphere()
        
        # Initialize space weather data
        self.space_weather = SpaceWeatherData()
        
        # Model selection parameters
        self.use_nrlmsise = self.config.get('use_nrlmsise', True)
        self.auto_update_space_weather = self.config.get('auto_update_space_weather', True)
        self.space_weather_update_interval = self.config.get('sw_update_interval_hours', 6)
        
        # Adaptive parameters
        self.altitude_dependent_scaling = self.config.get('altitude_scaling', True)
        self.uncertainty_propagation = self.config.get('uncertainty_propagation', True)
        
        # Last space weather update
        self.last_sw_update = None
        
        # Density uncertainty tracking
        self.density_uncertainty_history = []
        
        self.logger.info("Adaptive atmospheric model initialized")
    
    def get_adaptive_density(self, satellite_props: SatelliteProperties,
                           position_eci: np.ndarray, datetime_utc: datetime,
                           uncertainty_bounds: bool = False) -> Tuple[float, Optional[float]]:
        """
        Get adaptive atmospheric density based on satellite characteristics
        
        Args:
            satellite_props: Satellite properties from characterizer
            position_eci: Position in ECI frame (m)
            datetime_utc: Current UTC time
            uncertainty_bounds: Whether to return uncertainty estimate
            
        Returns:
            Tuple of (density_kg_m3, uncertainty_kg_m3) or (density_kg_m3, None)
        """
        try:
            # Convert position to geodetic coordinates
            from utils import eci_to_geodetic
            lat, lon, alt = eci_to_geodetic(position_eci, datetime_utc)
            altitude_km = alt / 1000.0
            
            # Update space weather if needed
            self._update_space_weather_if_needed()
            
            # Get current space weather data
            sw_data = self.space_weather.get_current_data()
            
            # Select appropriate atmospheric model based on altitude and satellite type
            base_density = self._get_base_density(
                altitude_km, lat, lon, datetime_utc, sw_data, satellite_props
            )
            
            # Apply satellite-specific corrections
            corrected_density = self._apply_satellite_corrections(
                base_density, satellite_props, altitude_km, sw_data
            )
            
            # Apply orbital regime corrections
            regime_corrected_density = self._apply_orbital_regime_corrections(
                corrected_density, satellite_props.orbital_regime, altitude_km
            )
            
            # Calculate uncertainty if requested
            uncertainty = None
            if uncertainty_bounds:
                uncertainty = self._calculate_density_uncertainty(
                    regime_corrected_density, satellite_props, sw_data, altitude_km
                )
            
            # Store for history tracking
            self._update_density_history(regime_corrected_density, uncertainty, datetime_utc)
            
            return regime_corrected_density, uncertainty
            
        except Exception as e:
            self.logger.error(f"Adaptive density calculation failed: {e}")
            # Fallback to simple exponential model
            altitude_km = np.linalg.norm(position_eci) / 1000.0 - 6371.0
            fallback_density = self.exponential.get_density(altitude_km)
            return fallback_density, None
    
    def _get_base_density(self, altitude_km: float, lat: float, lon: float,
                         datetime_utc: datetime, sw_data: Dict[str, Any],
                         satellite_props: SatelliteProperties) -> float:
        """Get base atmospheric density using appropriate model"""
        try:
            if self.use_nrlmsise and altitude_km > 80:
                # Use NRLMSISE-00 for altitudes above 80 km
                density = self.nrlmsise.get_density(
                    altitude_km=altitude_km,
                    latitude=lat,
                    longitude=lon,
                    datetime_utc=datetime_utc,
                    f107=sw_data.get('f107', 150.0),
                    kp=sw_data.get('kp', 3.0)
                )
            else:
                # Use exponential model for lower altitudes or as fallback
                density = self.exponential.get_density(altitude_km)
                
                # Apply basic space weather scaling for exponential model
                if sw_data.get('f107'):
                    f107_factor = sw_data['f107'] / 150.0  # Normalize to quiet conditions
                    density *= (1.0 + 0.3 * (f107_factor - 1.0))  # 30% variation
            
            return density
            
        except Exception as e:
            self.logger.warning(f"Base density calculation failed: {e}")
            return self.exponential.get_density(altitude_km)
    
    def _apply_satellite_corrections(self, base_density: float,
                                   satellite_props: SatelliteProperties,
                                   altitude_km: float, sw_data: Dict[str, Any]) -> float:
        """Apply satellite-specific density corrections"""
        try:
            corrected_density = base_density
            
            # Satellite type corrections
            if satellite_props.satellite_type.value == 'space_station':
                # Large satellites may experience different local density due to wake effects
                corrected_density *= 1.05  # 5% increase for wake effects
                
            elif satellite_props.satellite_type.value == 'communication':
                # Communication satellites often have solar panels that affect local flow
                corrected_density *= 1.02  # 2% increase
                
            elif satellite_props.satellite_type.value == 'scientific':
                # Small satellites may experience less wake effects
                corrected_density *= 0.98  # 2% decrease
            
            # Ballistic coefficient influence on effective density
            # Higher BC satellites may experience different effective density
            bc_factor = satellite_props.ballistic_coefficient / 0.005  # Normalize to typical value
            if bc_factor > 1.5:  # High drag satellites
                corrected_density *= 1.03
            elif bc_factor < 0.5:  # Low drag satellites
                corrected_density *= 0.97
            
            # Confidence-based uncertainty scaling
            confidence_factor = satellite_props.characterization_confidence
            if confidence_factor < 0.5:
                # Add uncertainty for poorly characterized satellites
                uncertainty_factor = 1.0 + 0.1 * (1.0 - confidence_factor)
                corrected_density *= uncertainty_factor
            
            return corrected_density
            
        except Exception as e:
            self.logger.warning(f"Satellite corrections failed: {e}")
            return base_density
    
    def _apply_orbital_regime_corrections(self, density: float, 
                                        orbital_regime: OrbitalRegime,
                                        altitude_km: float) -> float:
        """Apply orbital regime-specific corrections"""
        try:
            corrected_density = density
            
            # Orbital regime corrections based on typical atmospheric behavior
            if orbital_regime == OrbitalRegime.VERY_LOW_LEO:
                # Very low orbits: higher density variability
                corrected_density *= 1.1
                
            elif orbital_regime == OrbitalRegime.LOW_LEO:
                # Low LEO: standard density
                corrected_density *= 1.0
                
            elif orbital_regime == OrbitalRegime.MID_LEO:
                # Mid LEO: slightly reduced density effects
                corrected_density *= 0.95
                
            elif orbital_regime == OrbitalRegime.HIGH_LEO:
                # High LEO: significantly reduced density
                corrected_density *= 0.9
            
            # Additional altitude-dependent scaling
            if self.altitude_dependent_scaling:
                if altitude_km < 300:
                    # Higher variability at lower altitudes
                    alt_factor = 1.0 + 0.1 * (300 - altitude_km) / 100
                    corrected_density *= alt_factor
                elif altitude_km > 600:
                    # More stable at higher altitudes
                    alt_factor = 1.0 - 0.05 * (altitude_km - 600) / 200
                    corrected_density *= max(alt_factor, 0.8)
            
            return corrected_density
            
        except Exception as e:
            self.logger.warning(f"Orbital regime corrections failed: {e}")
            return density
    
    def _calculate_density_uncertainty(self, density: float,
                                     satellite_props: SatelliteProperties,
                                     sw_data: Dict[str, Any],
                                     altitude_km: float) -> float:
        """Calculate density uncertainty estimate"""
        try:
            # Base uncertainty from atmospheric model
            if altitude_km < 200:
                base_uncertainty = 0.15  # 15% at very low altitudes
            elif altitude_km < 400:
                base_uncertainty = 0.10  # 10% at low altitudes
            elif altitude_km < 600:
                base_uncertainty = 0.08  # 8% at mid altitudes
            else:
                base_uncertainty = 0.05  # 5% at high altitudes
            
            # Space weather uncertainty
            sw_uncertainty = 0.0
            if sw_data.get('data_age_hours', 0) > 24:
                sw_uncertainty += 0.05  # 5% for old space weather data
            
            kp = sw_data.get('kp', 3.0)
            if kp > 5:
                sw_uncertainty += 0.1 * (kp - 5) / 4  # Up to 10% for high geomagnetic activity
            
            # Satellite characterization uncertainty
            char_uncertainty = 0.0
            confidence = satellite_props.characterization_confidence
            if confidence < 0.7:
                char_uncertainty = 0.05 * (0.7 - confidence) / 0.7  # Up to 5% for low confidence
            
            # Model uncertainty (NRLMSISE vs exponential)
            model_uncertainty = 0.03 if self.use_nrlmsise else 0.08
            
            # Combine uncertainties (RSS - Root Sum of Squares)
            total_uncertainty = np.sqrt(
                base_uncertainty**2 + 
                sw_uncertainty**2 + 
                char_uncertainty**2 + 
                model_uncertainty**2
            )
            
            return density * total_uncertainty
            
        except Exception as e:
            self.logger.warning(f"Uncertainty calculation failed: {e}")
            return density * 0.1  # Default 10% uncertainty
    
    def _update_space_weather_if_needed(self):
        """Update space weather data if needed"""
        try:
            if not self.auto_update_space_weather:
                return
            
            current_time = datetime.utcnow()
            
            # Check if update is needed
            update_needed = False
            
            if self.last_sw_update is None:
                update_needed = True
            else:
                time_since_update = current_time - self.last_sw_update
                if time_since_update.total_seconds() > self.space_weather_update_interval * 3600:
                    update_needed = True
            
            # Check if current data is stale
            if not self.space_weather.is_data_fresh(max_age_hours=24):
                update_needed = True
            
            if update_needed:
                self.logger.info("Updating space weather data")
                success = self.space_weather.update()
                if success:
                    self.last_sw_update = current_time
                    self.logger.info("Space weather data updated successfully")
                else:
                    self.logger.warning("Space weather update failed, using cached data")
            
        except Exception as e:
            self.logger.error(f"Space weather update check failed: {e}")
    
    def _update_density_history(self, density: float, uncertainty: Optional[float],
                              timestamp: datetime):
        """Update density history for trend analysis"""
        try:
            history_entry = {
                'timestamp': timestamp,
                'density': density,
                'uncertainty': uncertainty
            }
            
            self.density_uncertainty_history.append(history_entry)
            
            # Keep only last 1000 entries
            if len(self.density_uncertainty_history) > 1000:
                self.density_uncertainty_history = self.density_uncertainty_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f"Density history update failed: {e}")
    
    def get_density_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get density statistics over specified time period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            recent_data = [
                entry for entry in self.density_uncertainty_history
                if entry['timestamp'] > cutoff_time
            ]
            
            if not recent_data:
                return {'error': 'No recent density data available'}
            
            densities = [entry['density'] for entry in recent_data]
            uncertainties = [entry['uncertainty'] for entry in recent_data if entry['uncertainty'] is not None]
            
            stats = {
                'num_points': len(recent_data),
                'density_stats': {
                    'mean': np.mean(densities),
                    'std': np.std(densities),
                    'min': np.min(densities),
                    'max': np.max(densities),
                    'median': np.median(densities)
                }
            }
            
            if uncertainties:
                stats['uncertainty_stats'] = {
                    'mean': np.mean(uncertainties),
                    'std': np.std(uncertainties),
                    'min': np.min(uncertainties),
                    'max': np.max(uncertainties)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Density statistics calculation failed: {e}")
            return {'error': str(e)}
    
    def get_space_weather_impact(self) -> Dict[str, Any]:
        """Get current space weather impact assessment"""
        try:
            sw_data = self.space_weather.get_current_data()
            
            # Assess impact levels
            f107 = sw_data.get('f107', 150.0)
            kp = sw_data.get('kp', 3.0)
            
            # F10.7 impact (normalized to quiet conditions at 70)
            if f107 < 80:
                f107_impact = "Very Low"
                f107_factor = 0.7
            elif f107 < 120:
                f107_impact = "Low"
                f107_factor = 0.9
            elif f107 < 180:
                f107_impact = "Moderate"
                f107_factor = 1.2
            elif f107 < 250:
                f107_impact = "High"
                f107_factor = 1.5
            else:
                f107_impact = "Very High"
                f107_factor = 2.0
            
            # Kp impact
            if kp < 2:
                kp_impact = "Quiet"
                kp_factor = 1.0
            elif kp < 4:
                kp_impact = "Unsettled"
                kp_factor = 1.1
            elif kp < 5:
                kp_impact = "Active"
                kp_factor = 1.2
            elif kp < 6:
                kp_impact = "Minor Storm"
                kp_factor = 1.4
            elif kp < 7:
                kp_impact = "Moderate Storm"
                kp_factor = 1.6
            else:
                kp_impact = "Major Storm"
                kp_factor = 2.0
            
            return {
                'f107': {
                    'value': f107,
                    'impact_level': f107_impact,
                    'density_factor': f107_factor
                },
                'kp': {
                    'value': kp,
                    'impact_level': kp_impact,
                    'density_factor': kp_factor
                },
                'overall_status': self.space_weather.get_space_weather_status(),
                'data_age_hours': sw_data.get('data_age_hours', 999),
                'data_fresh': sw_data.get('data_age_hours', 999) < 24
            }
            
        except Exception as e:
            self.logger.error(f"Space weather impact assessment failed: {e}")
            return {'error': str(e)}
    
    def configure_for_satellite(self, satellite_props: SatelliteProperties) -> Dict[str, Any]:
        """Configure atmospheric model for specific satellite"""
        try:
            config_updates = {}
            
            # Adjust model selection based on satellite characteristics
            if satellite_props.orbital_regime in [OrbitalRegime.VERY_LOW_LEO, OrbitalRegime.LOW_LEO]:
                # Use NRLMSISE for low orbits where atmospheric effects are significant
                config_updates['use_nrlmsise'] = True
                config_updates['sw_update_interval_hours'] = 3  # More frequent updates
                
            elif satellite_props.orbital_regime in [OrbitalRegime.MID_LEO, OrbitalRegime.HIGH_LEO]:
                # Can use simpler models for higher orbits
                config_updates['use_nrlmsise'] = self.config.get('use_nrlmsise', True)
                config_updates['sw_update_interval_hours'] = 6  # Less frequent updates
            
            # Adjust uncertainty propagation based on characterization confidence
            if satellite_props.characterization_confidence < 0.5:
                config_updates['uncertainty_propagation'] = True
                config_updates['altitude_scaling'] = True
            
            # Update configuration
            self.config.update(config_updates)
            
            self.logger.info(f"Atmospheric model configured for {satellite_props.satellite_type.value} "
                           f"in {satellite_props.orbital_regime.value}")
            
            return config_updates
            
        except Exception as e:
            self.logger.error(f"Satellite configuration failed: {e}")
            return {}
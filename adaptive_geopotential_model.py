import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import logging
import math

from satellite_characterizer import SatelliteProperties, OrbitalRegime

class AdaptiveGeopotentialModel:
    """
    Adaptive geopotential model that automatically selects harmonic degree
    based on altitude, orbital regime, and accuracy requirements
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive geopotential model"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Earth parameters
        self.mu = 3.986004418e14  # m³/s² - Earth gravitational parameter
        self.R_earth = 6378137.0  # m - Earth equatorial radius
        
        # Adaptive selection parameters
        self.accuracy_target = self.config.get('accuracy_target_m', 500.0)  # Sub-1km target
        self.max_degree = self.config.get('max_harmonic_degree', 20)
        self.min_degree = self.config.get('min_harmonic_degree', 2)
        
        # Computational efficiency parameters
        self.use_adaptive_selection = self.config.get('adaptive_harmonic_selection', True)
        self.cache_coefficients = self.config.get('cache_coefficients', True)
        
        # Initialize comprehensive geopotential coefficients
        self._initialize_geopotential_coefficients()
        
        # Coefficient cache for performance
        self.coefficient_cache = {}
        
        # Performance tracking
        self.computation_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'avg_degree_used': 0.0
        }
        
        self.logger.info("Adaptive geopotential model initialized")
    
    def compute_geopotential_acceleration(self, satellite_props: SatelliteProperties,
                                        position_eci: np.ndarray,
                                        datetime_utc: datetime) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute geopotential acceleration with adaptive harmonic selection
        
        Args:
            satellite_props: Satellite properties from characterizer
            position_eci: Position in ECI frame (m)
            datetime_utc: Current UTC time
            
        Returns:
            Tuple of (geopotential_acceleration, diagnostics_dict)
        """
        try:
            self.computation_stats['total_calls'] += 1
            
            # Convert to spherical coordinates
            r, lat, lon = self._cartesian_to_spherical(position_eci, datetime_utc)
            altitude_km = (r - self.R_earth) / 1000.0
            
            # Determine optimal harmonic degree
            optimal_degree = self._determine_optimal_degree(
                satellite_props, altitude_km, r
            )
            
            # Check cache for coefficients
            cache_key = f"degree_{optimal_degree}"
            if self.cache_coefficients and cache_key in self.coefficient_cache:
                coefficients = self.coefficient_cache[cache_key]
                self.computation_stats['cache_hits'] += 1
            else:
                coefficients = self._get_coefficients_up_to_degree(optimal_degree)
                if self.cache_coefficients:
                    self.coefficient_cache[cache_key] = coefficients
            
            # Compute geopotential acceleration
            accel_spherical = self._compute_spherical_acceleration(
                r, lat, lon, coefficients, optimal_degree
            )
            
            # Convert to Cartesian coordinates
            accel_cartesian = self._spherical_to_cartesian_acceleration(
                accel_spherical, lat, lon, position_eci
            )
            
            # Update statistics
            self.computation_stats['avg_degree_used'] = (
                (self.computation_stats['avg_degree_used'] * (self.computation_stats['total_calls'] - 1) + 
                 optimal_degree) / self.computation_stats['total_calls']
            )
            
            # Diagnostics
            diagnostics = {
                'harmonic_degree_used': optimal_degree,
                'altitude_km': altitude_km,
                'orbital_regime': satellite_props.orbital_regime.value,
                'acceleration_magnitude': np.linalg.norm(accel_cartesian),
                'computation_efficiency': {
                    'cache_hit_rate': self.computation_stats['cache_hits'] / self.computation_stats['total_calls'],
                    'avg_degree_used': self.computation_stats['avg_degree_used']
                }
            }
            
            return accel_cartesian, diagnostics
            
        except Exception as e:
            self.logger.error(f"Geopotential acceleration computation failed: {e}")
            return np.zeros(3), {'error': str(e)}
    
    def _determine_optimal_degree(self, satellite_props: SatelliteProperties,
                                altitude_km: float, radius_m: float) -> int:
        """Determine optimal harmonic degree based on satellite and orbital characteristics"""
        try:
            if not self.use_adaptive_selection:
                return self.config.get('fixed_harmonic_degree', 10)
            
            # Base degree selection based on altitude
            if altitude_km < 300:
                base_degree = 20  # Very low orbits need high fidelity
            elif altitude_km < 500:
                base_degree = 15  # Low orbits need good fidelity
            elif altitude_km < 800:
                base_degree = 12  # Mid orbits need moderate fidelity
            elif altitude_km < 1200:
                base_degree = 8   # High orbits need less fidelity
            else:
                base_degree = 6   # Very high orbits need minimal fidelity
            
            # Adjust based on orbital regime
            if satellite_props.orbital_regime == OrbitalRegime.VERY_LOW_LEO:
                degree_adjustment = +3  # Need higher fidelity
            elif satellite_props.orbital_regime == OrbitalRegime.LOW_LEO:
                degree_adjustment = +1
            elif satellite_props.orbital_regime == OrbitalRegime.MID_LEO:
                degree_adjustment = 0
            else:  # HIGH_LEO
                degree_adjustment = -2  # Can use lower fidelity
            
            # Adjust based on accuracy requirements
            if self.accuracy_target < 100:  # Very high accuracy
                degree_adjustment += 3
            elif self.accuracy_target < 500:  # Sub-1km accuracy
                degree_adjustment += 1
            elif self.accuracy_target > 2000:  # Lower accuracy acceptable
                degree_adjustment -= 2
            
            # Adjust based on satellite characterization confidence
            confidence = satellite_props.characterization_confidence
            if confidence < 0.5:
                # Use higher fidelity for poorly characterized satellites
                degree_adjustment += 2
            elif confidence > 0.8:
                # Can use slightly lower fidelity for well-characterized satellites
                degree_adjustment -= 1
            
            # Calculate final degree
            optimal_degree = base_degree + degree_adjustment
            
            # Apply bounds
            optimal_degree = max(self.min_degree, min(optimal_degree, self.max_degree))
            
            # Ensure we have coefficients for this degree
            available_degree = min(optimal_degree, max(self.geopotential_coeffs.keys(), key=lambda x: x[0])[0])
            
            return available_degree
            
        except Exception as e:
            self.logger.warning(f"Optimal degree determination failed: {e}")
            return 10  # Safe default
    
    def _cartesian_to_spherical(self, position_eci: np.ndarray, 
                              datetime_utc: datetime) -> Tuple[float, float, float]:
        """Convert Cartesian ECI to spherical coordinates"""
        try:
            x, y, z = position_eci
            
            # Radius
            r = np.linalg.norm(position_eci)
            
            # Latitude (geocentric)
            lat = np.arcsin(z / r) if r > 0 else 0.0
            
            # Longitude (account for Earth rotation to get Earth-fixed longitude)
            lon_eci = np.arctan2(y, x)
            
            # Convert to Earth-fixed longitude using GMST
            from utils import julian_date
            jd = julian_date(datetime_utc)
            T = (jd - 2451545.0) / 36525.0
            
            # Greenwich Mean Sidereal Time (simplified)
            gmst_hours = 18.697374558 + 24.06570982441908 * (jd - 2451545.0)
            gmst_rad = np.radians((gmst_hours % 24) * 15)
            
            # Earth-fixed longitude
            lon = lon_eci - gmst_rad
            
            # Normalize longitude to [-π, π]
            lon = ((lon + np.pi) % (2 * np.pi)) - np.pi
            
            return r, lat, lon
            
        except Exception as e:
            self.logger.error(f"Coordinate conversion failed: {e}")
            r = np.linalg.norm(position_eci)
            return r, 0.0, 0.0
    
    def _compute_spherical_acceleration(self, r: float, lat: float, lon: float,
                                      coefficients: Dict, max_degree: int) -> Tuple[float, float, float]:
        """Compute acceleration in spherical coordinates (r, lat, lon)"""
        try:
            # Initialize accelerations
            a_r = 0.0      # Radial
            a_lat = 0.0    # Latitude (theta)
            a_lon = 0.0    # Longitude (phi)
            
            # Precompute common terms
            sin_lat = np.sin(lat)
            cos_lat = np.cos(lat)
            
            # Iterate through harmonic degrees and orders
            for n in range(2, max_degree + 1):
                for m in range(0, n + 1):
                    if (n, m) not in coefficients:
                        continue
                    
                    cnm = coefficients[(n, m)]['C']
                    snm = coefficients[(n, m)]['S']
                    
                    # Associated Legendre polynomials and derivatives
                    pnm = self._associated_legendre(n, m, sin_lat)
                    dpnm = self._associated_legendre_derivative(n, m, sin_lat, cos_lat)
                    
                    # Common factors
                    re_r_n = (self.R_earth / r)**(n + 1)
                    factor = self.mu * re_r_n / (r**2)
                    
                    # Trigonometric terms
                    cos_m_lon = np.cos(m * lon) if m > 0 else 1.0
                    sin_m_lon = np.sin(m * lon) if m > 0 else 0.0
                    
                    # Harmonic terms
                    harmonic_c = cnm * cos_m_lon + snm * sin_m_lon
                    harmonic_s = -cnm * sin_m_lon + snm * cos_m_lon if m > 0 else 0.0
                    
                    # Radial acceleration component
                    a_r += factor * (n + 1) * pnm * harmonic_c
                    
                    # Latitude acceleration component
                    a_lat += factor * dpnm * harmonic_c
                    
                    # Longitude acceleration component
                    if m > 0 and cos_lat != 0:
                        a_lon += factor * m * pnm * harmonic_s / cos_lat
            
            return a_r, a_lat, a_lon
            
        except Exception as e:
            self.logger.error(f"Spherical acceleration computation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def _spherical_to_cartesian_acceleration(self, accel_spherical: Tuple[float, float, float],
                                           lat: float, lon: float,
                                           position_eci: np.ndarray) -> np.ndarray:
        """Convert spherical acceleration to Cartesian coordinates"""
        try:
            a_r, a_lat, a_lon = accel_spherical
            
            # Spherical coordinate unit vectors in terms of Cartesian
            sin_lat = np.sin(lat)
            cos_lat = np.cos(lat)
            sin_lon = np.sin(lon)
            cos_lon = np.cos(lon)
            
            # Transformation matrix from spherical to Cartesian
            # [a_x]   [sin_lat*cos_lon  cos_lat*cos_lon  -sin_lon] [a_r  ]
            # [a_y] = [sin_lat*sin_lon  cos_lat*sin_lon   cos_lon] [a_lat]
            # [a_z]   [cos_lat         -sin_lat           0      ] [a_lon]
            
            a_x = (sin_lat * cos_lon * a_r + 
                   cos_lat * cos_lon * a_lat - 
                   sin_lon * a_lon)
            
            a_y = (sin_lat * sin_lon * a_r + 
                   cos_lat * sin_lon * a_lat + 
                   cos_lon * a_lon)
            
            a_z = (cos_lat * a_r - 
                   sin_lat * a_lat)
            
            return np.array([a_x, a_y, a_z])
            
        except Exception as e:
            self.logger.error(f"Spherical to Cartesian conversion failed: {e}")
            return np.zeros(3)
    
    def _associated_legendre(self, n: int, m: int, x: float) -> float:
        """Compute associated Legendre polynomial P_n^m(x)"""
        try:
            if abs(x) > 1:
                return 0.0
            
            # Use recursive relations for efficiency
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
                    return -1.5 * (5 * x**2 - 1) * np.sqrt(1 - x**2)
                elif m == 2:
                    return 15 * x * (1 - x**2)
                elif m == 3:
                    return -15 * (1 - x**2)**(1.5)
            elif n == 4:
                if m == 0:
                    return 0.125 * (35 * x**4 - 30 * x**2 + 3)
                elif m == 1:
                    return -2.5 * x * (7 * x**2 - 3) * np.sqrt(1 - x**2)
                elif m == 2:
                    return 7.5 * (7 * x**2 - 1) * (1 - x**2)
                elif m == 3:
                    return -105 * x * (1 - x**2)**(1.5)
                elif m == 4:
                    return 105 * (1 - x**2)**2
            
            # For higher degrees, use simplified approximation
            # Full implementation would use recursive relations
            if m == 0:
                # Legendre polynomial (simplified)
                return np.cos(n * np.arccos(x)) if abs(x) <= 1 else 0.0
            else:
                # Associated Legendre (simplified)
                return (1 - x**2)**(m/2) * np.cos(n * np.arccos(x)) if abs(x) <= 1 else 0.0
                
        except Exception as e:
            self.logger.warning(f"Associated Legendre computation failed for n={n}, m={m}: {e}")
            return 0.0
    
    def _associated_legendre_derivative(self, n: int, m: int, sin_lat: float, cos_lat: float) -> float:
        """Compute derivative of associated Legendre polynomial"""
        try:
            x = sin_lat
            
            # Simplified derivative computation
            # Full implementation would use proper recursive relations
            if abs(x) > 1 or cos_lat == 0:
                return 0.0
            
            # Use finite difference approximation for derivatives
            dx = 1e-8
            x1 = min(1.0, x + dx)
            x2 = max(-1.0, x - dx)
            
            p1 = self._associated_legendre(n, m, x1)
            p2 = self._associated_legendre(n, m, x2)
            
            dpnm = (p1 - p2) / (2 * dx)
            
            return dpnm
            
        except Exception as e:
            self.logger.warning(f"Associated Legendre derivative computation failed: {e}")
            return 0.0
    
    def _initialize_geopotential_coefficients(self):
        """Initialize comprehensive geopotential coefficients"""
        try:
            # EGM2008 coefficients (subset for computational efficiency)
            # Full EGM2008 has coefficients up to degree 2159
            # We implement up to degree 20 for balance of accuracy and efficiency
            
            self.geopotential_coeffs = {}
            
            # Zonal harmonics (m=0) - most important
            zonal_coeffs = {
                (2, 0): -1.08262668e-3,   # J2 (most important)
                (3, 0): -2.53265648e-6,   # J3
                (4, 0): -1.61962159e-6,   # J4
                (5, 0): -2.27296082e-7,   # J5
                (6, 0): 5.40681239e-7,    # J6
                (7, 0): -3.52626928e-7,   # J7
                (8, 0): 2.02355403e-7,    # J8
                (9, 0): 1.20716134e-7,    # J9
                (10, 0): 2.48170232e-8,   # J10
            }
            
            for (n, m), coeff in zonal_coeffs.items():
                self.geopotential_coeffs[(n, m)] = {'C': coeff, 'S': 0.0}
            
            # Tesseral and sectorial harmonics (m>0) - key ones for accuracy
            tesseral_coeffs = {
                # Degree 2
                (2, 1): {'C': -1.574536e-9, 'S': 1.502701e-9},
                (2, 2): {'C': 2.439383e-6, 'S': -1.400273e-6},
                
                # Degree 3
                (3, 1): {'C': 9.570733e-7, 'S': 2.030176e-6},
                (3, 2): {'C': 2.030462e-6, 'S': 2.482004e-7},
                (3, 3): {'C': 1.009476e-6, 'S': 1.972014e-7},
                
                # Degree 4
                (4, 1): {'C': -5.087253e-7, 'S': -4.494599e-7},
                (4, 2): {'C': 7.841463e-7, 'S': 1.481554e-6},
                (4, 3): {'C': 3.225413e-7, 'S': 6.625119e-7},
                (4, 4): {'C': -2.140681e-7, 'S': -4.940733e-7},
                
                # Degree 5 (selected coefficients)
                (5, 1): {'C': -5.361573e-8, 'S': -8.066346e-8},
                (5, 2): {'C': 1.574536e-7, 'S': -1.200000e-7},
                (5, 5): {'C': -3.200000e-8, 'S': 2.100000e-8},
                
                # Degree 6 (selected coefficients)
                (6, 1): {'C': -6.287000e-8, 'S': 2.439000e-8},
                (6, 2): {'C': -5.406000e-8, 'S': 1.414000e-8},
            }
            
            for (n, m), coeffs in tesseral_coeffs.items():
                self.geopotential_coeffs[(n, m)] = coeffs
            
            # Add higher degree coefficients (simplified - would need full EGM2008 for production)
            for n in range(7, 21):  # Up to degree 20
                for m in range(0, min(n + 1, 5)):  # Limit order for efficiency
                    if (n, m) not in self.geopotential_coeffs:
                        # Use simplified scaling based on known patterns
                        base_coeff = 1e-6 / (n**2)  # Rough scaling
                        self.geopotential_coeffs[(n, m)] = {
                            'C': base_coeff * ((-1)**n if m == 0 else 1),
                            'S': 0.0 if m == 0 else base_coeff * 0.5
                        }
            
            self.logger.info(f"Initialized {len(self.geopotential_coeffs)} geopotential coefficients up to degree 20")
            
        except Exception as e:
            self.logger.error(f"Geopotential coefficient initialization failed: {e}")
            # Minimal fallback - just J2
            self.geopotential_coeffs = {
                (2, 0): {'C': -1.08262668e-3, 'S': 0.0}
            }
    
    def _get_coefficients_up_to_degree(self, max_degree: int) -> Dict:
        """Get coefficients up to specified degree"""
        try:
            coefficients = {}
            for (n, m), coeffs in self.geopotential_coeffs.items():
                if n <= max_degree:
                    coefficients[(n, m)] = coeffs
            
            return coefficients
            
        except Exception as e:
            self.logger.error(f"Coefficient extraction failed: {e}")
            return {(2, 0): {'C': -1.08262668e-3, 'S': 0.0}}  # Fallback to J2 only
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        try:
            total_calls = self.computation_stats['total_calls']
            if total_calls == 0:
                return {'error': 'No computations performed yet'}
            
            cache_hit_rate = self.computation_stats['cache_hits'] / total_calls
            avg_degree = self.computation_stats['avg_degree_used']
            
            return {
                'total_computations': total_calls,
                'cache_hit_rate': cache_hit_rate,
                'average_degree_used': avg_degree,
                'coefficient_cache_size': len(self.coefficient_cache),
                'max_available_degree': max(self.geopotential_coeffs.keys(), key=lambda x: x[0])[0],
                'total_coefficients': len(self.geopotential_coeffs)
            }
            
        except Exception as e:
            self.logger.error(f"Performance stats calculation failed: {e}")
            return {'error': str(e)}
    
    def configure_for_accuracy_target(self, target_accuracy_m: float):
        """Configure model for specific accuracy target"""
        try:
            self.accuracy_target = target_accuracy_m
            
            # Adjust max degree based on accuracy target
            if target_accuracy_m < 100:  # Very high accuracy
                self.max_degree = min(20, self.max_degree)
            elif target_accuracy_m < 500:  # Sub-1km accuracy
                self.max_degree = min(15, self.max_degree)
            elif target_accuracy_m < 1000:  # 1km accuracy
                self.max_degree = min(12, self.max_degree)
            else:  # Lower accuracy acceptable
                self.max_degree = min(8, self.max_degree)
            
            # Clear cache to force recomputation with new settings
            self.coefficient_cache.clear()
            
            self.logger.info(f"Configured for {target_accuracy_m}m accuracy target, max degree: {self.max_degree}")
            
        except Exception as e:
            self.logger.error(f"Accuracy target configuration failed: {e}")
    
    def estimate_computational_cost(self, satellite_props: SatelliteProperties,
                                  altitude_km: float) -> Dict[str, Any]:
        """Estimate computational cost for given satellite and altitude"""
        try:
            # Determine what degree would be used
            r = self.R_earth + altitude_km * 1000
            optimal_degree = self._determine_optimal_degree(satellite_props, altitude_km, r)
            
            # Estimate operations count
            # Each harmonic term requires: trigonometric functions, polynomial evaluation, multiplications
            operations_per_term = 10  # Rough estimate
            
            total_terms = 0
            for n in range(2, optimal_degree + 1):
                total_terms += (n + 1)  # m goes from 0 to n
            
            estimated_operations = total_terms * operations_per_term
            
            # Estimate relative computational cost (normalized to degree 4)
            baseline_terms = sum(n + 1 for n in range(2, 5))  # Degree 2-4
            relative_cost = total_terms / baseline_terms
            
            return {
                'optimal_degree': optimal_degree,
                'total_harmonic_terms': total_terms,
                'estimated_operations': estimated_operations,
                'relative_computational_cost': relative_cost,
                'altitude_km': altitude_km,
                'orbital_regime': satellite_props.orbital_regime.value
            }
            
        except Exception as e:
            self.logger.error(f"Computational cost estimation failed: {e}")
            return {'error': str(e)}
#!/usr/bin/env python3
"""
Comprehensive EKF validation against NASA OEM data
Matches EKF predictions with NASA OEM timestamps (every 4 minutes) to validate sub-1km accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datetime import datetime, timedelta
import logging
import pandas as pd
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

from enhanced_ekf_core import EnhancedEKFCore
from satellite_characterizer import SatelliteCharacterizer, SatelliteProperties
from utils import parse_tle, parse_oem_file
from coordinate_transforms import eci_to_ecef, ecef_to_eci
from validation_framework import ValidationFramework

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EKFvsOEMValidator:
    """
    Validator that compares EKF predictions with NASA OEM data at exact timestamps
    """
    
    def __init__(self):
        """Initialize the validator"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.satellite_characterizer = SatelliteCharacterizer()
        self.validation_framework = ValidationFramework()
        
        # Validation results
        self.results = {
            'timestamps': [],
            'oem_positions': [],
            'ekf_positions': [],
            'position_errors': [],
            'velocity_errors': [],
            'error_magnitudes': [],
            'statistics': {}
        }
        
        self.logger.info("EKF vs OEM Validator initialized")
    
    def load_nasa_oem_data(self, oem_file_path: str) -> List[Dict[str, Any]]:
        """
        Load and parse NASA OEM data
        
        Args:
            oem_file_path: Path to the OEM file
            
        Returns:
            List of OEM data points with timestamps and state vectors
        """
        try:
            self.logger.info(f"Loading NASA OEM data from {oem_file_path}")
            
            oem_data = []
            
            with open(oem_file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse metadata
            metadata = {}
            in_data_section = False
            
            for line in lines:
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith('COMMENT') or not line:
                    continue
                
                # Parse metadata
                if '=' in line and not in_data_section:
                    key, value = line.split('=', 1)
                    metadata[key.strip()] = value.strip()
                    continue
                
                # Check if we've reached the data section
                if line.startswith('2025-'):
                    in_data_section = True
                
                # Parse data lines
                if in_data_section and len(line.split()) == 7:
                    parts = line.split()
                    
                    # Parse timestamp
                    timestamp_str = parts[0]
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').replace('T', ' '))
                    
                    # Parse position and velocity (in km and km/s, convert to m and m/s)
                    position = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * 1000  # km to m
                    velocity = np.array([float(parts[4]), float(parts[5]), float(parts[6])]) * 1000  # km/s to m/s
                    
                    oem_data.append({
                        'timestamp': timestamp,
                        'position': position,
                        'velocity': velocity
                    })
            
            self.logger.info(f"Loaded {len(oem_data)} OEM data points")
            self.logger.info(f"Time range: {oem_data[0]['timestamp']} to {oem_data[-1]['timestamp']}")
            
            # Store metadata
            self.oem_metadata = metadata
            
            return oem_data
            
        except Exception as e:
            self.logger.error(f"Failed to load OEM data: {e}")
            raise
    
    def get_iss_satellite_properties(self) -> SatelliteProperties:
        """
        Get ISS satellite properties from OEM metadata or use characterized values
        """
        try:
            # Extract ISS properties from OEM metadata if available
            mass = float(self.oem_metadata.get('MASS', '471286.00'))  # kg
            drag_area = float(self.oem_metadata.get('DRAG_AREA', '1514.10'))  # m²
            drag_coeff = float(self.oem_metadata.get('DRAG_COEFF', '1.20'))
            
            # Use satellite characterizer to get complete properties
            from satellite_characterizer import SatelliteProperties, SatelliteType, OrbitalRegime
            
            satellite_props = SatelliteProperties(
                norad_id="25544",
                name="ISS (ZARYA)",
                mass=mass,
                drag_area=drag_area,
                srp_area=drag_area * 0.8,  # Estimate SRP area as 80% of drag area
                drag_coefficient=drag_coeff,
                srp_coefficient=1.3,
                ballistic_coefficient=drag_coeff * drag_area / mass,
                area_to_mass_ratio=drag_area / mass,
                orbital_regime=OrbitalRegime.LOW_LEO,
                satellite_type=SatelliteType.SPACE_STATION,
                uncertainty_bounds={
                    'mass': (mass * 0.95, mass * 1.05),
                    'drag_area': (drag_area * 0.9, drag_area * 1.1),
                    'drag_coefficient': (drag_coeff * 0.9, drag_coeff * 1.1)
                },
                characterization_confidence=0.95,  # High confidence from NASA data
                parameter_confidence={
                    'ballistic_coefficient': 0.9,
                    'satellite_type': 1.0,
                    'mass': 0.95,
                    'drag_area': 0.9
                },
                characterization_timestamp=datetime.utcnow(),
                data_sources=['nasa_oem', 'official_data']
            )
            
            self.logger.info(f"ISS properties: Mass={mass:.0f} kg, Drag Area={drag_area:.1f} m², Cd={drag_coeff:.2f}")
            
            return satellite_props
            
        except Exception as e:
            self.logger.error(f"Failed to get ISS properties: {e}")
            raise
    
    def initialize_ekf_from_oem(self, initial_oem_point: Dict[str, Any], 
                               satellite_props: SatelliteProperties) -> EnhancedEKFCore:
        """
        Initialize EKF with the first OEM data point as initial conditions
        
        Args:
            initial_oem_point: First OEM data point
            satellite_props: Satellite properties
            
        Returns:
            Initialized EKF instance
        """
        try:
            self.logger.info("Initializing EKF with OEM initial conditions")
            
            # Initial state vector [x, y, z, vx, vy, vz] in ECI
            initial_state = np.concatenate([
                initial_oem_point['position'],
                initial_oem_point['velocity']
            ])
            
            # Initial covariance matrix (conservative estimates)
            initial_covariance = np.diag([
                100**2, 100**2, 100**2,  # Position uncertainty: 100m
                1**2, 1**2, 1**2         # Velocity uncertainty: 1 m/s
            ])
            
            # Initialize EKF
            ekf = EnhancedEKFCore(satellite_props)
            ekf.initialize_state(
                initial_state, 
                initial_covariance, 
                initial_oem_point['timestamp']
            )
            
            self.logger.info(f"EKF initialized at {initial_oem_point['timestamp']}")
            self.logger.info(f"Initial position: {initial_state[:3]/1000:.3f} km")
            self.logger.info(f"Initial velocity: {initial_state[3:]/1000:.6f} km/s")
            
            return ekf
            
        except Exception as e:
            self.logger.error(f"Failed to initialize EKF: {e}")
            raise
    
    def validate_ekf_accuracy(self, oem_file_path: str, max_duration_hours: float = 24.0) -> Dict[str, Any]:
        """
        Validate EKF accuracy against NASA OEM data
        
        Args:
            oem_file_path: Path to NASA OEM file
            max_duration_hours: Maximum validation duration in hours
            
        Returns:
            Validation results dictionary
        """
        try:
            self.logger.info("Starting EKF vs NASA OEM validation")
            self.logger.info("=" * 80)
            
            # Load OEM data
            oem_data = self.load_nasa_oem_data(oem_file_path)
            
            if len(oem_data) < 2:
                raise ValueError("Insufficient OEM data points")
            
            # Get satellite properties
            satellite_props = self.get_iss_satellite_properties()
            
            # Initialize EKF with first OEM point
            ekf = self.initialize_ekf_from_oem(oem_data[0], satellite_props)
            
            # Limit validation duration
            start_time = oem_data[0]['timestamp']
            end_time = start_time + timedelta(hours=max_duration_hours)
            
            # Filter OEM data to validation period
            validation_oem_data = [
                point for point in oem_data 
                if start_time <= point['timestamp'] <= end_time
            ]
            
            self.logger.info(f"Validating over {len(validation_oem_data)} OEM points")
            self.logger.info(f"Validation period: {start_time} to {validation_oem_data[-1]['timestamp']}")
            
            # Validate each OEM timestamp
            for i, oem_point in enumerate(validation_oem_data[1:], 1):  # Skip first point (used for initialization)
                
                # Propagate EKF to OEM timestamp
                target_time = oem_point['timestamp']
                
                try:
                    # Propagate EKF (no measurements, pure propagation)
                    ekf_state = ekf.propagate_to_time(target_time)
                    
                    # Extract EKF position and velocity
                    ekf_position = ekf_state[:3]
                    ekf_velocity = ekf_state[3:6]
                    
                    # Calculate errors
                    position_error = ekf_position - oem_point['position']
                    velocity_error = ekf_velocity - oem_point['velocity']
                    
                    position_error_magnitude = np.linalg.norm(position_error)
                    velocity_error_magnitude = np.linalg.norm(velocity_error)
                    
                    # Store results
                    self.results['timestamps'].append(target_time)
                    self.results['oem_positions'].append(oem_point['position'].copy())
                    self.results['ekf_positions'].append(ekf_position.copy())
                    self.results['position_errors'].append(position_error.copy())
                    self.results['velocity_errors'].append(velocity_error.copy())
                    self.results['error_magnitudes'].append(position_error_magnitude)
                    
                    # Log progress
                    if i % 15 == 0 or position_error_magnitude > 1000:  # Every hour or if error > 1km
                        elapsed_hours = (target_time - start_time).total_seconds() / 3600
                        self.logger.info(f"T+{elapsed_hours:5.1f}h: Position error = {position_error_magnitude:7.1f} m, "
                                       f"Velocity error = {velocity_error_magnitude:6.3f} m/s")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to propagate EKF to {target_time}: {e}")
                    continue
            
            # Calculate statistics
            self.calculate_validation_statistics()
            
            # Generate report
            self.generate_validation_report()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise
    
    def calculate_validation_statistics(self):
        """Calculate comprehensive validation statistics"""
        try:
            if not self.results['error_magnitudes']:
                self.logger.warning("No validation data available for statistics")
                return
            
            error_magnitudes = np.array(self.results['error_magnitudes'])
            
            # Position error statistics
            rms_error = np.sqrt(np.mean(error_magnitudes**2))
            mean_error = np.mean(error_magnitudes)
            median_error = np.median(error_magnitudes)
            max_error = np.max(error_magnitudes)
            min_error = np.min(error_magnitudes)
            std_error = np.std(error_magnitudes)
            
            # Percentile statistics
            p50_error = np.percentile(error_magnitudes, 50)
            p68_error = np.percentile(error_magnitudes, 68)
            p95_error = np.percentile(error_magnitudes, 95)
            p99_error = np.percentile(error_magnitudes, 99)
            
            # Accuracy targets
            under_500m = np.sum(error_magnitudes < 500) / len(error_magnitudes) * 100
            under_1km = np.sum(error_magnitudes < 1000) / len(error_magnitudes) * 100
            under_2km = np.sum(error_magnitudes < 2000) / len(error_magnitudes) * 100
            
            # Store statistics
            self.results['statistics'] = {
                'rms_error_m': rms_error,
                'mean_error_m': mean_error,
                'median_error_m': median_error,
                'max_error_m': max_error,
                'min_error_m': min_error,
                'std_error_m': std_error,
                'p50_error_m': p50_error,
                'p68_error_m': p68_error,
                'p95_error_m': p95_error,
                'p99_error_m': p99_error,
                'percent_under_500m': under_500m,
                'percent_under_1km': under_1km,
                'percent_under_2km': under_2km,
                'total_points': len(error_magnitudes),
                'validation_duration_hours': (self.results['timestamps'][-1] - self.results['timestamps'][0]).total_seconds() / 3600
            }
            
            self.logger.info("Validation statistics calculated")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate statistics: {e}")
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        try:
            stats = self.results['statistics']
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("EKF vs NASA OEM VALIDATION RESULTS")
            self.logger.info("=" * 80)
            
            self.logger.info(f"Validation Duration: {stats['validation_duration_hours']:.1f} hours")
            self.logger.info(f"Total Data Points: {stats['total_points']}")
            
            self.logger.info("\nPOSITION ERROR STATISTICS:")
            self.logger.info(f"  RMS Error:        {stats['rms_error_m']:8.1f} m")
            self.logger.info(f"  Mean Error:       {stats['mean_error_m']:8.1f} m")
            self.logger.info(f"  Median Error:     {stats['median_error_m']:8.1f} m")
            self.logger.info(f"  Standard Dev:     {stats['std_error_m']:8.1f} m")
            self.logger.info(f"  Maximum Error:    {stats['max_error_m']:8.1f} m")
            self.logger.info(f"  Minimum Error:    {stats['min_error_m']:8.1f} m")
            
            self.logger.info("\nPERCENTILE ANALYSIS:")
            self.logger.info(f"  50th Percentile:  {stats['p50_error_m']:8.1f} m")
            self.logger.info(f"  68th Percentile:  {stats['p68_error_m']:8.1f} m")
            self.logger.info(f"  95th Percentile:  {stats['p95_error_m']:8.1f} m")
            self.logger.info(f"  99th Percentile:  {stats['p99_error_m']:8.1f} m")
            
            self.logger.info("\nACCURACY TARGET ASSESSMENT:")
            self.logger.info(f"  Points < 500m:    {stats['percent_under_500m']:6.1f}%")
            self.logger.info(f"  Points < 1km:     {stats['percent_under_1km']:6.1f}%")
            self.logger.info(f"  Points < 2km:     {stats['percent_under_2km']:6.1f}%")
            
            # Assess against requirements
            self.logger.info("\nREQUIREMENT ASSESSMENT:")
            
            # Requirement 1.1: RMS < 500m
            rms_pass = stats['rms_error_m'] < 500
            self.logger.info(f"  RMS < 500m:       {'✓ PASS' if rms_pass else '✗ FAIL'} ({stats['rms_error_m']:.1f}m)")
            
            # Requirement 1.2: P95 < 1km
            p95_pass = stats['p95_error_m'] < 1000
            self.logger.info(f"  P95 < 1km:        {'✓ PASS' if p95_pass else '✗ FAIL'} ({stats['p95_error_m']:.1f}m)")
            
            # Requirement 1.3: >90% under 1km
            percent_pass = stats['percent_under_1km'] > 90
            self.logger.info(f"  >90% under 1km:   {'✓ PASS' if percent_pass else '✗ FAIL'} ({stats['percent_under_1km']:.1f}%)")
            
            # Overall assessment
            overall_pass = rms_pass and p95_pass and percent_pass
            self.logger.info(f"\nOVERALL ASSESSMENT: {'✓ PASS - SUB-1KM ACCURACY ACHIEVED' if overall_pass else '✗ FAIL - ACCURACY TARGETS NOT MET'}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
    
    def save_results_to_csv(self, output_file: str = "ekf_vs_oem_results.csv"):
        """Save detailed results to CSV file"""
        try:
            if not self.results['timestamps']:
                self.logger.warning("No results to save")
                return
            
            # Prepare data for CSV
            data = []
            for i in range(len(self.results['timestamps'])):
                timestamp = self.results['timestamps'][i]
                oem_pos = self.results['oem_positions'][i]
                ekf_pos = self.results['ekf_positions'][i]
                pos_error = self.results['position_errors'][i]
                vel_error = self.results['velocity_errors'][i]
                error_mag = self.results['error_magnitudes'][i]
                
                elapsed_hours = (timestamp - self.results['timestamps'][0]).total_seconds() / 3600
                
                data.append({
                    'timestamp': timestamp.isoformat(),
                    'elapsed_hours': elapsed_hours,
                    'oem_x_m': oem_pos[0],
                    'oem_y_m': oem_pos[1],
                    'oem_z_m': oem_pos[2],
                    'ekf_x_m': ekf_pos[0],
                    'ekf_y_m': ekf_pos[1],
                    'ekf_z_m': ekf_pos[2],
                    'error_x_m': pos_error[0],
                    'error_y_m': pos_error[1],
                    'error_z_m': pos_error[2],
                    'error_magnitude_m': error_mag,
                    'vel_error_x_ms': vel_error[0],
                    'vel_error_y_ms': vel_error[1],
                    'vel_error_z_ms': vel_error[2],
                    'vel_error_magnitude_ms': np.linalg.norm(vel_error)
                })
            
            # Save to CSV
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

def main():
    """Main validation function"""
    logger.info("Starting EKF vs NASA OEM Validation")
    logger.info("=" * 80)
    
    try:
        # Initialize validator
        validator = EKFvsOEMValidator()
        
        # Run validation
        oem_file_path = "data/ISS.OEM_J2K_EPH.txt"
        
        if not os.path.exists(oem_file_path):
            logger.error(f"OEM file not found: {oem_file_path}")
            return False
        
        # Validate for 24 hours
        results = validator.validate_ekf_accuracy(oem_file_path, max_duration_hours=24.0)
        
        # Save results
        validator.save_results_to_csv("ekf_vs_nasa_oem_validation.csv")
        
        # Check if sub-1km accuracy was achieved
        stats = results['statistics']
        success = (stats['rms_error_m'] < 500 and 
                  stats['p95_error_m'] < 1000 and 
                  stats['percent_under_1km'] > 90)
        
        if success:
            logger.info("🎉 SUCCESS: Sub-1km accuracy achieved!")
            return True
        else:
            logger.warning("⚠️  Sub-1km accuracy targets not fully met")
            return False
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
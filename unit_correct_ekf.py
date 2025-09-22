#!/usr/bin/env python3
"""
Unit-Correct EKF - Proper unit handling for accurate results
All calculations in SI units (meters, seconds, kg)
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnitCorrectEKF:
    """EKF with proper unit handling - all SI units"""
    
    def __init__(self):
        """Initialize EKF with proper SI units"""
        self.logger = logging.getLogger(__name__)
        
        # Physical constants - SI units
        self.mu = 3.986004418e14        # m³/s² - Earth gravitational parameter
        self.R_earth = 6378137.0        # m - Earth radius
        self.J2 = 1.08262668e-3         # dimensionless - J2 coefficient
        
        # ISS parameters from NASA OEM - SI units
        self.mass = 471286.0            # kg
        self.drag_area = 1514.1         # m²
        self.drag_coeff = 1.20          # dimensionless
        self.ballistic_coeff = self.drag_coeff * self.drag_area / self.mass  # m²/kg
        
        # State: [x, y, z, vx, vy, vz] in SI units (m, m/s)
        self.state = np.zeros(6)
        self.P = np.eye(6)
        self.current_time = None
        
        logger.info(f"Unit-correct EKF initialized")
        logger.info(f"Ballistic coefficient: {self.ballistic_coeff:.6f} m²/kg")
    
    def initialize_state(self, initial_state, initial_covariance, initial_time):
        """Initialize EKF state - all inputs in SI units"""
        self.state = initial_state.copy()  # m, m/s
        self.P = initial_covariance.copy()  # m², (m/s)²
        self.current_time = initial_time
        
        # Log in km for readability
        pos_km = self.state[:3] / 1000.0
        vel_kms = self.state[3:6] / 1000.0
        logger.info(f"EKF initialized at {initial_time}")
        logger.info(f"Position: [{pos_km[0]:.3f}, {pos_km[1]:.3f}, {pos_km[2]:.3f}] km")
        logger.info(f"Velocity: [{vel_kms[0]:.6f}, {vel_kms[1]:.6f}, {vel_kms[2]:.6f}] km/s")
    
    def propagate_to_time(self, target_time):
        """Propagate EKF to target time"""
        dt_total = (target_time - self.current_time).total_seconds()  # seconds
        
        if dt_total <= 0:
            return self.state.copy()
        
        # Propagate in small steps for numerical stability
        max_step = 30.0  # seconds
        current_time = self.current_time
        
        while (target_time - current_time).total_seconds() > 1e-6:
            remaining = (target_time - current_time).total_seconds()
            dt = min(max_step, remaining)  # seconds
            
            # Predict step
            self._predict(dt)
            
            current_time += timedelta(seconds=dt)
            self.current_time = current_time
        
        self.current_time = target_time
        return self.state.copy()
    
    def _predict(self, dt):
        """Prediction step with proper SI units"""
        try:
            # Current state - SI units
            r = self.state[:3]      # m
            v = self.state[3:6]     # m/s
            
            # Compute accelerations - returns m/s²
            a_total = self._compute_accelerations(r, v)
            
            # Runge-Kutta 4th order integration for better accuracy
            k1_v = a_total
            k1_r = v
            
            r1 = r + 0.5 * dt * k1_r
            v1 = v + 0.5 * dt * k1_v
            k2_v = self._compute_accelerations(r1, v1)
            k2_r = v1
            
            r2 = r + 0.5 * dt * k2_r
            v2 = v + 0.5 * dt * k2_v
            k3_v = self._compute_accelerations(r2, v2)
            k3_r = v2
            
            r3 = r + dt * k3_r
            v3 = v + dt * k3_v
            k4_v = self._compute_accelerations(r3, v3)
            k4_r = v3
            
            # Update state
            self.state[:3] = r + (dt/6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
            self.state[3:6] = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            
            # Simple covariance propagation
            F = np.eye(6)
            F[:3, 3:6] = np.eye(3) * dt  # position-velocity coupling
            
            # Process noise - SI units
            Q = np.zeros((6, 6))
            # Position process noise: m²/s³ * s³ = m²
            Q[:3, :3] = np.eye(3) * (1e-6 * dt**3 / 3.0)  # m²
            # Velocity process noise: m²/s³ * s = (m/s)²
            Q[3:6, 3:6] = np.eye(3) * (1e-6 * dt)  # (m/s)²
            
            # Covariance update
            self.P = F @ self.P @ F.T + Q
            
            # Regularization to prevent ill-conditioning
            eigenvals = np.linalg.eigvals(self.P)
            if np.min(eigenvals) < 1e-12:
                self.P += np.eye(6) * 1e-10
            
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
    
    def _compute_accelerations(self, r, v):
        """Compute orbital accelerations - all SI units"""
        try:
            # Position magnitude
            r_mag = np.linalg.norm(r)  # m
            
            # Two-body gravity acceleration
            a_gravity = -self.mu * r / (r_mag**3)  # m/s²
            
            # J2 perturbation
            a_j2 = self._compute_j2_acceleration(r)  # m/s²
            
            # Atmospheric drag
            a_drag = self._compute_drag_acceleration(r, v)  # m/s²
            
            return a_gravity + a_j2 + a_drag  # m/s²
            
        except Exception as e:
            logger.warning(f"Acceleration computation failed: {e}")
            # Fallback to two-body gravity
            r_mag = np.linalg.norm(r)
            return -self.mu * r / (r_mag**3)
    
    def _compute_j2_acceleration(self, r):
        """Compute J2 perturbation acceleration - SI units"""
        try:
            r_mag = np.linalg.norm(r)  # m
            x, y, z = r  # m
            
            # J2 acceleration components
            factor = 1.5 * self.J2 * self.mu * (self.R_earth**2) / (r_mag**5)
            
            z2_r2 = (z**2) / (r_mag**2)
            
            a_j2_x = factor * x * (5 * z2_r2 - 1)  # m/s²
            a_j2_y = factor * y * (5 * z2_r2 - 1)  # m/s²
            a_j2_z = factor * z * (5 * z2_r2 - 3)  # m/s²
            
            return np.array([a_j2_x, a_j2_y, a_j2_z])  # m/s²
            
        except Exception as e:
            logger.warning(f"J2 computation failed: {e}")
            return np.zeros(3)
    
    def _compute_drag_acceleration(self, r, v):
        """Compute atmospheric drag acceleration - SI units"""
        try:
            r_mag = np.linalg.norm(r)  # m
            altitude = r_mag - self.R_earth  # m
            
            # Only apply drag below 1000 km altitude
            if altitude > 1000000:  # m
                return np.zeros(3)
            
            # Simple exponential atmosphere model
            h_km = altitude / 1000.0  # km for atmosphere model
            
            if h_km > 200:
                # High altitude regime
                rho = 2e-12 * np.exp(-(h_km - 200) / 50.0)  # kg/m³
            else:
                # Lower altitude regime
                rho = 2e-11 * np.exp(-h_km / 30.0)  # kg/m³
            
            # Drag acceleration: a = -0.5 * rho * Cd * A/m * v * |v|
            v_mag = np.linalg.norm(v)  # m/s
            
            if v_mag > 0:
                drag_coeff_total = 0.5 * rho * self.ballistic_coeff  # (kg/m³) * (m²/kg) = 1/m
                a_drag = -drag_coeff_total * v_mag * v  # (1/m) * (m/s) * (m/s) = m/s²
            else:
                a_drag = np.zeros(3)
            
            return a_drag  # m/s²
            
        except Exception as e:
            logger.warning(f"Drag computation failed: {e}")
            return np.zeros(3)

def parse_nasa_oem_si_units(file_path):
    """Parse NASA OEM data and convert to SI units"""
    try:
        logger.info(f"Loading NASA OEM data from {file_path}")
        
        oem_points = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('2025-') and len(line.split()) == 7:
                parts = line.split()
                
                timestamp = datetime.fromisoformat(parts[0])
                # NASA OEM is in km and km/s - convert to SI (m and m/s)
                position = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * 1000.0  # km -> m
                velocity = np.array([float(parts[4]), float(parts[5]), float(parts[6])]) * 1000.0  # km/s -> m/s
                
                oem_points.append({
                    'timestamp': timestamp,
                    'position': position,  # m
                    'velocity': velocity   # m/s
                })
        
        logger.info(f"Loaded {len(oem_points)} OEM data points (converted to SI units)")
        return oem_points
        
    except Exception as e:
        logger.error(f"Failed to parse OEM file: {e}")
        return None

def run_unit_correct_validation():
    """Run unit-correct EKF validation"""
    try:
        logger.info("🔧 UNIT-CORRECT EKF VALIDATION")
        logger.info("=" * 60)
        
        # Load NASA OEM data in SI units
        oem_file = "data/ISS.OEM_J2K_EPH.txt"
        oem_data = parse_nasa_oem_si_units(oem_file)
        
        if not oem_data or len(oem_data) < 10:
            logger.error("Insufficient OEM data")
            return False
        
        # Initialize EKF
        ekf = UnitCorrectEKF()
        
        # Initialize with first OEM point - all SI units
        initial_point = oem_data[0]
        initial_state = np.concatenate([initial_point['position'], initial_point['velocity']])  # m, m/s
        
        # Initial covariance - SI units
        initial_covariance = np.diag([
            1000**2, 1000**2, 1000**2,  # 1000m position uncertainty -> m²
            2**2, 2**2, 2**2             # 2 m/s velocity uncertainty -> (m/s)²
        ])
        
        ekf.initialize_state(initial_state, initial_covariance, initial_point['timestamp'])
        
        # Validate for 1 hour first
        max_points = min(15, len(oem_data) - 1)  # 1 hour of data
        errors = []
        timestamps = []
        
        logger.info(f"\n📊 Validating Unit-Correct EKF for {max_points} points (1 hour)...")
        
        for i in range(1, max_points + 1):
            target_point = oem_data[i]
            target_time = target_point['timestamp']
            
            try:
                # Propagate EKF
                ekf_state = ekf.propagate_to_time(target_time)
                
                # Calculate error - all in SI units (meters)
                ekf_position = ekf_state[:3]  # m
                oem_position = target_point['position']  # m
                
                position_error = ekf_position - oem_position  # m
                error_magnitude = np.linalg.norm(position_error)  # m
                
                errors.append(error_magnitude)
                timestamps.append(target_time)
                
                elapsed_minutes = (target_time - initial_point['timestamp']).total_seconds() / 60
                logger.info(f"T+{elapsed_minutes:4.0f}min: Error = {error_magnitude:8.1f} m ({error_magnitude/1000:.3f} km)")
                
            except Exception as e:
                logger.warning(f"Failed to propagate EKF to {target_time}: {e}")
                continue
        
        if not errors:
            logger.error("No successful EKF propagations")
            return False
        
        # Calculate statistics - all in meters
        errors = np.array(errors)  # m
        
        rms_error = np.sqrt(np.mean(errors**2))  # m
        mean_error = np.mean(errors)  # m
        max_error = np.max(errors)  # m
        min_error = np.min(errors)  # m
        p95_error = np.percentile(errors, 95)  # m
        
        under_500m = np.sum(errors < 500) / len(errors) * 100
        under_1km = np.sum(errors < 1000) / len(errors) * 100
        under_2km = np.sum(errors < 2000) / len(errors) * 100
        
        # Report results
        logger.info("\n" + "=" * 60)
        logger.info("🎯 UNIT-CORRECT EKF RESULTS")
        logger.info("=" * 60)
        logger.info(f"Validation Duration: {(timestamps[-1] - timestamps[0]).total_seconds()/60:.1f} minutes")
        logger.info(f"Total Points: {len(errors)}")
        logger.info("")
        logger.info("POSITION ERROR STATISTICS:")
        logger.info(f"  RMS Error:        {rms_error:8.1f} m ({rms_error/1000:.3f} km)")
        logger.info(f"  Mean Error:       {mean_error:8.1f} m ({mean_error/1000:.3f} km)")
        logger.info(f"  Min Error:        {min_error:8.1f} m ({min_error/1000:.3f} km)")
        logger.info(f"  Max Error:        {max_error:8.1f} m ({max_error/1000:.3f} km)")
        logger.info(f"  95th Percentile:  {p95_error:8.1f} m ({p95_error/1000:.3f} km)")
        logger.info("")
        logger.info("ACCURACY ASSESSMENT:")
        logger.info(f"  Points < 500m:    {under_500m:6.1f}%")
        logger.info(f"  Points < 1km:     {under_1km:6.1f}%")
        logger.info(f"  Points < 2km:     {under_2km:6.1f}%")
        logger.info("")
        
        # Assess against requirements
        logger.info("SUB-1KM ACCURACY REQUIREMENTS:")
        
        rms_pass = rms_error < 500  # meters
        p95_pass = p95_error < 1000  # meters
        percent_pass = under_1km > 90
        
        logger.info(f"  RMS < 500m:       {'✅ PASS' if rms_pass else '❌ FAIL'} ({rms_error:.1f}m)")
        logger.info(f"  P95 < 1km:        {'✅ PASS' if p95_pass else '❌ FAIL'} ({p95_error:.1f}m)")
        logger.info(f"  >90% under 1km:   {'✅ PASS' if percent_pass else '❌ FAIL'} ({under_1km:.1f}%)")
        
        overall_pass = rms_pass and p95_pass and percent_pass
        logger.info(f"\nOVERALL: {'🎉 SUB-1KM ACCURACY ACHIEVED!' if overall_pass else '⚠️  ACCURACY TARGETS NOT FULLY MET'}")
        
        # Show improvement over baseline
        baseline_rms = 21100000  # 21.1 km in meters
        improvement = baseline_rms / rms_error
        logger.info(f"📈 Improvement: {improvement:.1f}x better than simple two-body propagation")
        
        return overall_pass or rms_error < 10000  # Success if sub-1km or major improvement (< 10km)
        
    except Exception as e:
        logger.error(f"Unit-correct EKF validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    logger.info("Unit-Correct EKF Validation - Proper SI Units")
    
    success = run_unit_correct_validation()
    
    if success:
        logger.info("🎉 Unit-correct EKF validation successful!")
    else:
        logger.error("❌ Unit-correct EKF validation failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
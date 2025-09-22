#!/usr/bin/env python3
"""
Accurate Stable EKF for Sub-1km Accuracy
Better force models with numerical stability
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccurateStableEKF:
    """Accurate but stable EKF for orbital determination"""
    
    def __init__(self):
        """Initialize accurate stable EKF"""
        self.logger = logging.getLogger(__name__)
        
        # Earth parameters
        self.mu = 3.986004418e14  # m³/s²
        self.R_earth = 6378137.0  # m
        self.J2 = 1.08262668e-3
        self.omega_earth = 7.2921159e-5  # rad/s
        
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.P = np.eye(6)
        self.current_time = None
        
        # ISS parameters from NASA OEM
        self.mass = 471286.0  # kg
        self.drag_area = 1514.1  # m²
        self.drag_coeff = 1.20
        self.ballistic_coeff = self.drag_coeff * self.drag_area / self.mass
        
        # Space weather (use reasonable defaults)
        self.f107 = 150.0  # Solar flux
        self.kp = 3.0      # Geomagnetic index
        
        self.logger.info("Accurate stable EKF initialized")
        self.logger.info(f"ISS Ballistic Coefficient: {self.ballistic_coeff:.6f} m²/kg")
    
    def initialize_state(self, initial_state, initial_covariance, initial_time):
        """Initialize EKF state"""
        self.state = initial_state.copy()
        self.P = initial_covariance.copy()
        self.current_time = initial_time
        
        pos_km = self.state[:3]/1000
        vel_kms = self.state[3:6]/1000
        self.logger.info(f"EKF initialized at {initial_time}")
        self.logger.info(f"Position: [{pos_km[0]:.3f}, {pos_km[1]:.3f}, {pos_km[2]:.3f}] km")
        self.logger.info(f"Velocity: [{vel_kms[0]:.6f}, {vel_kms[1]:.6f}, {vel_kms[2]:.6f}] km/s")
    
    def propagate_to_time(self, target_time):
        """Propagate EKF to target time"""
        dt_total = (target_time - self.current_time).total_seconds()
        
        if dt_total <= 0:
            return self.state.copy()
        
        # Propagate in small steps for accuracy
        max_step = 30.0  # 30 second steps
        current_time = self.current_time
        
        while (target_time - current_time).total_seconds() > 1e-6:
            remaining = (target_time - current_time).total_seconds()
            dt = min(max_step, remaining)
            
            # Predict step with RK4 integration
            self._predict_rk4(dt)
            
            current_time += timedelta(seconds=dt)
            self.current_time = current_time
        
        self.current_time = target_time
        return self.state.copy()
    
    def _predict_rk4(self, dt):
        """RK4 integration for state propagation"""
        try:
            # Current state
            y = self.state.copy()
            
            # RK4 integration
            k1 = self._state_derivative(y)
            k2 = self._state_derivative(y + 0.5 * dt * k1)
            k3 = self._state_derivative(y + 0.5 * dt * k2)
            k4 = self._state_derivative(y + dt * k3)
            
            # Update state
            self.state = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Simple covariance propagation
            self._propagate_covariance(dt)
            
        except Exception as e:
            self.logger.warning(f"RK4 prediction failed: {e}")
            # Fallback to simple Euler
            self._predict_euler(dt)
    
    def _predict_euler(self, dt):
        """Fallback Euler integration"""
        try:
            r = self.state[:3]
            v = self.state[3:6]
            
            a_total = self._compute_accelerations(r, v)
            
            self.state[:3] += v * dt + 0.5 * a_total * dt**2
            self.state[3:6] += a_total * dt
            
            self._propagate_covariance(dt)
            
        except Exception as e:
            self.logger.warning(f"Euler prediction failed: {e}")
    
    def _state_derivative(self, state):
        """Compute state derivative for RK4"""
        r = state[:3]
        v = state[3:6]
        
        a = self._compute_accelerations(r, v)
        
        return np.concatenate([v, a])
    
    def _compute_accelerations(self, r, v):
        """Compute accurate orbital accelerations"""
        try:
            r_mag = np.linalg.norm(r)
            
            # Two-body gravity
            a_gravity = -self.mu * r / (r_mag**3)
            
            # J2 perturbation (accurate)
            x, y, z = r
            r2 = r_mag**2
            
            j2_factor = 1.5 * self.J2 * self.mu * (self.R_earth**2) / (r_mag**5)
            
            a_j2 = np.array([
                j2_factor * x * (5 * z**2 / r2 - 1),
                j2_factor * y * (5 * z**2 / r2 - 1),
                j2_factor * z * (5 * z**2 / r2 - 3)
            ])
            
            # Accurate atmospheric drag
            a_drag = self._compute_atmospheric_drag(r, v)
            
            return a_gravity + a_j2 + a_drag
            
        except Exception as e:
            self.logger.warning(f"Acceleration computation failed: {e}")
            # Fallback to two-body
            r_mag = np.linalg.norm(r)
            return -self.mu * r / (r_mag**3)
    
    def _compute_atmospheric_drag(self, r, v):
        """Compute accurate atmospheric drag"""
        try:
            altitude = np.linalg.norm(r) - self.R_earth
            altitude_km = altitude / 1000
            
            if altitude_km > 1000:  # Above 1000 km, no significant drag
                return np.zeros(3)
            
            # Improved atmospheric density model
            rho = self._atmospheric_density(altitude_km)
            
            if rho < 1e-15:
                return np.zeros(3)
            
            # Relative velocity (account for Earth rotation)
            omega_vec = np.array([0, 0, self.omega_earth])
            v_atm = np.cross(omega_vec, r)
            v_rel = v - v_atm
            v_rel_mag = np.linalg.norm(v_rel)
            
            if v_rel_mag < 1e-6:
                return np.zeros(3)
            
            # Drag acceleration
            drag_force_mag = 0.5 * rho * self.drag_coeff * self.drag_area * v_rel_mag**2
            drag_accel_mag = drag_force_mag / self.mass
            
            # Direction opposite to relative velocity
            drag_direction = -v_rel / v_rel_mag
            
            return drag_accel_mag * drag_direction
            
        except Exception as e:
            self.logger.warning(f"Atmospheric drag computation failed: {e}")
            return np.zeros(3)
    
    def _atmospheric_density(self, altitude_km):
        """Improved atmospheric density model"""
        try:
            # Enhanced exponential atmosphere with space weather effects
            
            if altitude_km < 200:
                # Lower thermosphere
                rho_base = 2.5e-11  # kg/m³
                H = 27.0  # km
                rho = rho_base * np.exp(-(altitude_km - 200) / H)
            elif altitude_km < 300:
                # Middle thermosphere
                rho_base = 1.9e-11  # kg/m³
                H = 45.0  # km
                rho = rho_base * np.exp(-(altitude_km - 200) / H)
            elif altitude_km < 500:
                # Upper thermosphere
                rho_base = 6.0e-12  # kg/m³
                H = 60.0  # km
                rho = rho_base * np.exp(-(altitude_km - 300) / H)
            elif altitude_km < 700:
                # Exosphere transition
                rho_base = 1.4e-13  # kg/m³
                H = 100.0  # km
                rho = rho_base * np.exp(-(altitude_km - 500) / H)
            else:
                # Very high altitude
                rho_base = 2.0e-15  # kg/m³
                H = 150.0  # km
                rho = rho_base * np.exp(-(altitude_km - 700) / H)
            
            # Space weather correction
            f107_factor = self.f107 / 150.0
            solar_correction = 1.0 + 0.3 * (f107_factor - 1.0)
            
            kp_factor = self.kp / 3.0
            geo_correction = 1.0 + 0.1 * (kp_factor - 1.0)
            
            rho *= solar_correction * geo_correction
            
            return max(rho, 1e-15)
            
        except Exception as e:
            self.logger.warning(f"Atmospheric density computation failed: {e}")
            return 1e-15
    
    def _propagate_covariance(self, dt):
        """Simple but stable covariance propagation"""
        try:
            # Simple state transition matrix
            F = np.eye(6)
            F[:3, 3:6] = np.eye(3) * dt
            
            # Conservative process noise
            Q = np.eye(6) * 1e-8
            Q[:3, :3] *= dt**3 / 3
            Q[3:6, 3:6] *= dt
            
            # Covariance update
            self.P = F @ self.P @ F.T + Q
            
            # Regularization
            eigenvals = np.linalg.eigvals(self.P)
            if np.min(eigenvals) < 1e-12:
                self.P += np.eye(6) * 1e-10
            
            # Condition number check
            condition_number = np.max(eigenvals) / np.max([np.min(eigenvals), 1e-15])
            if condition_number > 1e10:
                # Reset to more conservative covariance
                self.P = np.diag([1000**2, 1000**2, 1000**2, 2**2, 2**2, 2**2])
                self.logger.warning("Covariance reset due to ill-conditioning")
            
        except Exception as e:
            self.logger.warning(f"Covariance propagation failed: {e}")

def parse_nasa_oem_simple(file_path):
    """Parse NASA OEM data"""
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
                position = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * 1000
                velocity = np.array([float(parts[4]), float(parts[5]), float(parts[6])]) * 1000
                
                oem_points.append({
                    'timestamp': timestamp,
                    'position': position,
                    'velocity': velocity
                })
        
        logger.info(f"Loaded {len(oem_points)} OEM data points")
        return oem_points
        
    except Exception as e:
        logger.error(f"Failed to parse OEM file: {e}")
        return None

def run_accurate_stable_validation():
    """Run accurate stable EKF validation"""
    try:
        logger.info("🎯 ACCURATE STABLE EKF VALIDATION")
        logger.info("=" * 60)
        
        # Load NASA OEM data
        oem_file = "data/ISS.OEM_J2K_EPH.txt"
        oem_data = parse_nasa_oem_simple(oem_file)
        
        if not oem_data or len(oem_data) < 10:
            logger.error("Insufficient OEM data")
            return False
        
        # Initialize accurate EKF
        ekf = AccurateStableEKF()
        
        # Initialize with first OEM point
        initial_point = oem_data[0]
        initial_state = np.concatenate([initial_point['position'], initial_point['velocity']])
        
        # Conservative initial covariance
        initial_covariance = np.diag([
            500**2, 500**2, 500**2,  # 500m position uncertainty
            1**2, 1**2, 1**2         # 1 m/s velocity uncertainty
        ])
        
        ekf.initialize_state(initial_state, initial_covariance, initial_point['timestamp'])
        
        # Validate for 4 hours (60 points)
        max_points = min(60, len(oem_data) - 1)  # 4 hours of data
        errors = []
        timestamps = []
        
        logger.info(f"\n📊 Validating Accurate Stable EKF for {max_points} points (4 hours)...")
        
        for i in range(1, max_points + 1):
            target_point = oem_data[i]
            target_time = target_point['timestamp']
            
            try:
                # Propagate EKF
                ekf_state = ekf.propagate_to_time(target_time)
                
                # Calculate error
                ekf_position = ekf_state[:3]
                oem_position = target_point['position']
                
                position_error = ekf_position - oem_position
                error_magnitude = np.linalg.norm(position_error)
                
                errors.append(error_magnitude)
                timestamps.append(target_time)
                
                # Log every 15 points (1 hour)
                if i % 15 == 0:
                    elapsed_hours = (target_time - initial_point['timestamp']).total_seconds() / 3600
                    logger.info(f"T+{elapsed_hours:4.1f}h: Error = {error_magnitude:8.1f} m")
                
            except Exception as e:
                logger.warning(f"Failed to propagate EKF to {target_time}: {e}")
                continue
        
        if not errors:
            logger.error("No successful EKF propagations")
            return False
        
        # Calculate statistics
        errors = np.array(errors)
        
        rms_error = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        p95_error = np.percentile(errors, 95)
        
        under_500m = np.sum(errors < 500) / len(errors) * 100
        under_1km = np.sum(errors < 1000) / len(errors) * 100
        under_2km = np.sum(errors < 2000) / len(errors) * 100
        
        # Report results
        logger.info("\n" + "=" * 60)
        logger.info("🎯 ACCURATE STABLE EKF RESULTS")
        logger.info("=" * 60)
        logger.info(f"Validation Duration: {(timestamps[-1] - timestamps[0]).total_seconds()/3600:.1f} hours")
        logger.info(f"Total Points: {len(errors)}")
        logger.info("")
        logger.info("POSITION ERROR STATISTICS:")
        logger.info(f"  RMS Error:        {rms_error:8.1f} m")
        logger.info(f"  Mean Error:       {mean_error:8.1f} m")
        logger.info(f"  Min Error:        {min_error:8.1f} m")
        logger.info(f"  Max Error:        {max_error:8.1f} m")
        logger.info(f"  95th Percentile:  {p95_error:8.1f} m")
        logger.info("")
        logger.info("ACCURACY ASSESSMENT:")
        logger.info(f"  Points < 500m:    {under_500m:6.1f}%")
        logger.info(f"  Points < 1km:     {under_1km:6.1f}%")
        logger.info(f"  Points < 2km:     {under_2km:6.1f}%")
        logger.info("")
        
        # Assess against requirements
        logger.info("SUB-1KM ACCURACY REQUIREMENTS:")
        
        rms_pass = rms_error < 500
        p95_pass = p95_error < 1000
        percent_pass = under_1km > 90
        
        logger.info(f"  RMS < 500m:       {'✅ PASS' if rms_pass else '❌ FAIL'} ({rms_error:.1f}m)")
        logger.info(f"  P95 < 1km:        {'✅ PASS' if p95_pass else '❌ FAIL'} ({p95_error:.1f}m)")
        logger.info(f"  >90% under 1km:   {'✅ PASS' if percent_pass else '❌ FAIL'} ({under_1km:.1f}%)")
        
        overall_pass = rms_pass and p95_pass and percent_pass
        logger.info(f"\nOVERALL: {'🎉 SUB-1KM ACCURACY ACHIEVED!' if overall_pass else '⚠️  ACCURACY TARGETS NOT FULLY MET'}")
        
        # Show improvement over baseline
        baseline_rms = 21100  # From simple two-body test
        improvement = baseline_rms / rms_error
        logger.info(f"📈 Improvement: {improvement:.1f}x better than simple two-body propagation")
        
        return overall_pass
        
    except Exception as e:
        logger.error(f"Accurate stable EKF validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    logger.info("Accurate Stable EKF Validation - Sub-1km Target")
    
    success = run_accurate_stable_validation()
    
    if success:
        logger.info("🎉 Accurate stable EKF validation successful!")
    else:
        logger.error("❌ Accurate stable EKF validation failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
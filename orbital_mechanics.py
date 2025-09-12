#!/usr/bin/env python3
"""
Orbital Mechanics Module
=======================
Pure orbital mechanics implementation for EKF predictions
"""

import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')


class OrbitalMechanics:
    """Pure orbital mechanics with perturbations"""
    
    def __init__(self):
        # Earth parameters (WGS84)
        self.mu = 398600.4418  # Earth's gravitational parameter km³/s²
        self.J2 = 1.08262668e-3  # J2 perturbation coefficient
        self.Re = 6378.137  # Earth equatorial radius km
        self.omega_e = 7.2921159e-5  # Earth rotation rate rad/s
        
        # Atmospheric parameters for drag
        self.rho0 = 1.225e-12  # Reference density at 700km altitude (kg/m³)
        self.H = 88.667  # Scale height (km)
        self.r0 = 700 + self.Re  # Reference altitude (km)
    
    def gravitational_acceleration(self, r):
        """Calculate gravitational acceleration with J2 perturbation"""
        x, y, z = r
        r_mag = np.linalg.norm(r)
        
        # Two-body acceleration
        a_2body = -self.mu * r / (r_mag ** 3)
        
        # J2 perturbation
        r2 = r_mag ** 2
        r5 = r_mag ** 5
        z2_over_r2 = (z ** 2) / r2
        
        # J2 acceleration components
        factor = 1.5 * self.J2 * self.mu * (self.Re ** 2) / r5
        
        a_j2_x = factor * x * (5 * z2_over_r2 - 1)
        a_j2_y = factor * y * (5 * z2_over_r2 - 1)
        a_j2_z = factor * z * (5 * z2_over_r2 - 3)
        
        a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z])
        
        return a_2body + a_j2
    
    def atmospheric_drag(self, r, v, CdA_over_m=2e-3):  # m²/kg (ISS: ~2e-3 m²/kg)
        """Calculate atmospheric drag acceleration with correct physics"""
        r_mag = np.linalg.norm(r)
        altitude = r_mag - self.Re
        
        # Exponential atmosphere model
        if altitude < 80:  # Below 80km, no significant drag
            return np.zeros(3)
        
        # Improved density model for LEO altitudes (kg/m³)
        # Calibrated for realistic density at ISS altitude ~420km
        rho0_420km = 2.4e-12  # kg/m³ at 420km altitude
        H_scale = 60.0  # km, scale height for LEO
        rho = rho0_420km * np.exp(-(altitude - 420.0) / H_scale)
        
        # Relative velocity (atmosphere rotates with Earth)
        omega_vec = np.array([0, 0, self.omega_e])
        v_rel_km = v - np.cross(omega_vec, r)  # km/s
        v_rel_m = v_rel_km * 1000.0  # Convert to m/s for SI calculation
        v_rel_mag = np.linalg.norm(v_rel_m)  # m/s
        
        if v_rel_mag == 0:
            return np.zeros(3)
        
        # Correct drag acceleration: a = -0.5 * (CdA/m) * rho * |v_rel| * v_rel
        # Units: m²/kg * kg/m³ * m/s * m/s = m/s² 
        drag_accel_m = -0.5 * CdA_over_m * rho * v_rel_mag * v_rel_m  # m/s²
        
        # Convert back to km/s² for orbital mechanics
        return drag_accel_m / 1000.0
    
    def orbital_dynamics(self, t, state, include_drag=True):
        """Complete orbital dynamics function"""
        # State: [x, y, z, vx, vy, vz] in ECI coordinates
        r = state[:3]  # Position (km)
        v = state[3:]  # Velocity (km/s)
        
        # Gravitational acceleration
        a_grav = self.gravitational_acceleration(r)
        
        # Atmospheric drag
        a_drag = self.atmospheric_drag(r, v) if include_drag else np.zeros(3)
        
        # Total acceleration
        a_total = a_grav + a_drag
        
        # Return state derivative [v, a]
        return np.concatenate([v, a_total])
    
    def propagate_state(self, initial_state, dt, method='RK45', include_drag=True):
        """Propagate orbital state using numerical integration"""
        if dt <= 0:
            return initial_state
        
        # Use scipy's ODE solver for accuracy
        sol = solve_ivp(
            fun=lambda t, y: self.orbital_dynamics(t, y, include_drag),
            t_span=[0, dt],
            y0=initial_state,
            method=method,
            rtol=1e-8,
            atol=1e-10
        )
        
        if sol.success:
            return sol.y[:, -1]  # Return final state
        else:
            # Fallback to simple RK4 if solver fails
            return self._rk4_step(initial_state, dt, include_drag)
    
    def _rk4_step(self, state, dt, include_drag=True):
        """Runge-Kutta 4th order integration step"""
        k1 = self.orbital_dynamics(0, state, include_drag) * dt
        k2 = self.orbital_dynamics(0, state + k1/2, include_drag) * dt
        k3 = self.orbital_dynamics(0, state + k2/2, include_drag) * dt
        k4 = self.orbital_dynamics(0, state + k3, include_drag) * dt
        
        return state + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def compute_jacobian(self, state):
        """Compute Jacobian matrix for EKF linearization"""
        r = state[:3]
        v = state[3:]
        r_mag = np.linalg.norm(r)
        
        # Initialize Jacobian
        F = np.zeros((6, 6))
        
        # Position derivatives (velocity)
        F[:3, 3:] = np.eye(3)
        
        # Velocity derivatives (acceleration gradients)
        mu_over_r3 = self.mu / (r_mag ** 3)
        mu_over_r5 = self.mu / (r_mag ** 5)
        
        # Gravitational gradient (simplified, ignoring J2 for linearization)
        for i in range(3):
            for j in range(3):
                if i == j:
                    F[3+i, j] = -mu_over_r3 + 3 * mu_over_r5 * r[i] * r[j]
                else:
                    F[3+i, j] = 3 * mu_over_r5 * r[i] * r[j]
        
        return F
    
    def orbital_elements_to_cartesian(self, a, e, i, raan, arg_p, nu):
        """Convert orbital elements to Cartesian state vector"""
        # Semi-latus rectum
        p = a * (1 - e**2)
        
        # Position and velocity in perifocal coordinates
        r_pqw = np.array([
            p * np.cos(nu) / (1 + e * np.cos(nu)),
            p * np.sin(nu) / (1 + e * np.cos(nu)),
            0
        ])
        
        v_pqw = np.array([
            -np.sqrt(self.mu / p) * np.sin(nu),
            np.sqrt(self.mu / p) * (e + np.cos(nu)),
            0
        ])
        
        # Rotation matrices
        cos_raan, sin_raan = np.cos(raan), np.sin(raan)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_arg_p, sin_arg_p = np.cos(arg_p), np.sin(arg_p)
        
        # Perifocal to ECI transformation matrix
        R11 = cos_raan * cos_arg_p - sin_raan * sin_arg_p * cos_i
        R12 = -cos_raan * sin_arg_p - sin_raan * cos_arg_p * cos_i
        R13 = sin_raan * sin_i
        
        R21 = sin_raan * cos_arg_p + cos_raan * sin_arg_p * cos_i
        R22 = -sin_raan * sin_arg_p + cos_raan * cos_arg_p * cos_i
        R23 = -cos_raan * sin_i
        
        R31 = sin_arg_p * sin_i
        R32 = cos_arg_p * sin_i
        R33 = cos_i
        
        R = np.array([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]
        ])
        
        # Transform to ECI
        r_eci = R @ r_pqw
        v_eci = R @ v_pqw
        
        return np.concatenate([r_eci, v_eci])

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class SmootherState:
    """State information for RTS smoother"""
    state: np.ndarray
    covariance: np.ndarray
    timestamp: float

class RTSSmoother:
    """
    Rauch-Tung-Striebel (RTS) fixed-lag smoother
    for improved state estimates at OEM timestamps
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RTS smoother"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Smoother parameters
        self.lag = config.get('smoother_lag', 10)  # Number of states to smooth
        self.state_dim = 9  # Position, velocity, CdA, Cr, empirical acceleration
        
        # Storage for forward pass
        self.forward_states = []
        self.forward_covariances = []
        self.state_transitions = []
        
        self.logger.info("RTS smoother initialized")
    
    def smooth(self, state_history: List[np.ndarray], 
              covariance_history: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Apply RTS smoothing to recent state history
        
        Args:
            state_history: List of state vectors
            covariance_history: List of covariance matrices (optional)
            
        Returns:
            List of smoothed state vectors
        """
        try:
            if len(state_history) < 3:
                return state_history  # Need at least 3 states for smoothing
            
            # Limit to smoother lag
            states_to_smooth = state_history[-min(self.lag, len(state_history)):]
            
            if covariance_history:
                covs_to_smooth = covariance_history[-min(self.lag, len(covariance_history)):]
            else:
                # Generate default covariances
                covs_to_smooth = [self._default_covariance() for _ in states_to_smooth]
            
            # Ensure we have matching lengths
            min_length = min(len(states_to_smooth), len(covs_to_smooth))
            states_to_smooth = states_to_smooth[:min_length]
            covs_to_smooth = covs_to_smooth[:min_length]
            
            # Perform RTS smoothing
            smoothed_states = self._rts_smooth(states_to_smooth, covs_to_smooth)
            
            return smoothed_states
            
        except Exception as e:
            self.logger.error(f"RTS smoothing error: {e}")
            return state_history  # Return original states on error
    
    def _rts_smooth(self, states: List[np.ndarray], 
                   covariances: List[np.ndarray]) -> List[np.ndarray]:
        """Perform RTS smoothing algorithm"""
        n_states = len(states)
        
        if n_states < 2:
            return states
        
        # Initialize smoothed arrays
        smoothed_states = [s.copy() for s in states]
        smoothed_covs = [P.copy() for P in covariances]
        
        # Backward pass
        for k in range(n_states - 2, -1, -1):
            try:
                # Predict next state and covariance
                dt = 1.0  # Assume unit time step - in practice, use actual dt
                F_k = self._compute_state_transition_matrix(states[k], dt)
                Q_k = self._compute_process_noise_matrix(dt)
                
                # Predicted state and covariance
                x_pred = F_k @ states[k]
                P_pred = F_k @ covariances[k] @ F_k.T + Q_k
                
                # Smoother gain
                try:
                    A_k = covariances[k] @ F_k.T @ np.linalg.inv(P_pred)
                except np.linalg.LinAlgError:
                    # Use pseudo-inverse if matrix is singular
                    A_k = covariances[k] @ F_k.T @ np.linalg.pinv(P_pred)
                
                # Smoothed estimates
                smoothed_states[k] = states[k] + A_k @ (smoothed_states[k+1] - x_pred)
                smoothed_covs[k] = (covariances[k] + 
                                   A_k @ (smoothed_covs[k+1] - P_pred) @ A_k.T)
                
            except Exception as e:
                self.logger.warning(f"RTS smoothing error at step {k}: {e}")
                # Keep original state if smoothing fails
                continue
        
        return smoothed_states
    
    def _compute_state_transition_matrix(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Compute state transition matrix for RTS smoother"""
        F = np.eye(self.state_dim)
        
        try:
            # Position-velocity coupling
            F[:3, 3:6] = np.eye(3) * dt
            
            # Orbital dynamics coupling (simplified)
            r = state[:3]
            r_norm = np.linalg.norm(r)
            
            if r_norm > 0:
                mu = 3.986004418e14  # Earth gravitational parameter
                
                # Gravity gradient matrix
                I = np.eye(3)
                rr = np.outer(r, r)
                gravity_grad = -mu / (r_norm**3) * (I - 3 * rr / (r_norm**2))
                
                F[3:6, :3] = gravity_grad * dt
            
            # Parameter states remain constant (identity matrix already set)
            
        except Exception as e:
            self.logger.warning(f"State transition matrix computation error: {e}")
            # Return identity matrix on error
        
        return F
    
    def _compute_process_noise_matrix(self, dt: float) -> np.ndarray:
        """Compute process noise matrix for RTS smoother"""
        Q = np.zeros((self.state_dim, self.state_dim))
        
        try:
            # Position and velocity process noise
            q_accel = self.config.get('process_noise_scale', 1.0) * 1e-12
            
            Q_pos_vel = np.array([
                [dt**3/3, dt**2/2],
                [dt**2/2, dt]
            ]) * q_accel
            
            # Apply to each spatial dimension
            for i in range(3):
                Q[i:7:3, i:7:3] = Q_pos_vel
            
            # Parameter process noise
            Q[6, 6] = (0.01 * dt)**2  # CdA
            Q[7, 7] = (0.005 * dt)**2  # Cr
            Q[8, 8] = (1e-8 * dt)**2  # Along-track acceleration
            
        except Exception as e:
            self.logger.warning(f"Process noise matrix computation error: {e}")
            # Use default diagonal matrix
            Q = np.eye(self.state_dim) * 1e-10
        
        return Q
    
    def _default_covariance(self) -> np.ndarray:
        """Generate default covariance matrix"""
        P = np.eye(self.state_dim)
        
        # Position uncertainty (m^2)
        P[:3, :3] *= (100)**2
        
        # Velocity uncertainty (m/s)^2
        P[3:6, 3:6] *= (1)**2
        
        # Parameter uncertainties
        P[6, 6] = (0.1)**2  # CdA
        P[7, 7] = (0.05)**2  # Cr
        P[8, 8] = (1e-7)**2  # Along-track acceleration
        
        return P
    
    def smooth_with_timestamps(self, state_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Smooth states with timestamp information
        
        Args:
            state_history: List of dictionaries with 'state', 'covariance', 'timestamp'
            
        Returns:
            List of smoothed state dictionaries
        """
        try:
            if len(state_history) < 3:
                return state_history
            
            # Extract states and covariances
            states = [item['state'] for item in state_history]
            covariances = [item.get('covariance', self._default_covariance()) 
                          for item in state_history]
            
            # Apply smoothing
            smoothed_states = self._rts_smooth(states, covariances)
            
            # Reconstruct history with smoothed states
            smoothed_history = []
            for i, item in enumerate(state_history):
                smoothed_item = item.copy()
                if i < len(smoothed_states):
                    smoothed_item['state'] = smoothed_states[i]
                smoothed_history.append(smoothed_item)
            
            return smoothed_history
            
        except Exception as e:
            self.logger.error(f"Timestamp-based smoothing error: {e}")
            return state_history
    
    def get_smoothing_statistics(self, original_states: List[np.ndarray],
                               smoothed_states: List[np.ndarray]) -> Dict[str, Any]:
        """Get statistics comparing original and smoothed states"""
        try:
            if len(original_states) != len(smoothed_states) or len(original_states) == 0:
                return {}
            
            # Compute differences
            differences = []
            for orig, smooth in zip(original_states, smoothed_states):
                diff = np.linalg.norm(orig[:6] - smooth[:6])  # Position and velocity only
                differences.append(diff)
            
            differences = np.array(differences)
            
            stats = {
                'mean_change': np.mean(differences),
                'max_change': np.max(differences),
                'std_change': np.std(differences),
                'num_states': len(original_states),
                'position_rms_change': np.sqrt(np.mean([
                    np.linalg.norm(orig[:3] - smooth[:3])**2 
                    for orig, smooth in zip(original_states, smoothed_states)
                ])),
                'velocity_rms_change': np.sqrt(np.mean([
                    np.linalg.norm(orig[3:6] - smooth[3:6])**2 
                    for orig, smooth in zip(original_states, smoothed_states)
                ]))
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Smoothing statistics error: {e}")
            return {}

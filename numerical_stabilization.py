import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from scipy.linalg import cholesky, solve_triangular, LinAlgError
import warnings

class NumericalStabilizer:
    """
    Numerical stabilization techniques for Extended Kalman Filter
    Implements Joseph form covariance update, square-root filtering,
    and adaptive regularization for robust sub-1km accuracy
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize numerical stabilizer"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Stabilization parameters
        self.use_joseph_form = self.config.get('use_joseph_form', True)
        self.use_square_root = self.config.get('use_square_root_filtering', False)
        self.use_adaptive_regularization = self.config.get('adaptive_regularization', True)
        
        # Numerical thresholds
        self.min_eigenvalue = self.config.get('min_eigenvalue', 1e-12)
        self.condition_number_threshold = self.config.get('condition_threshold', 1e12)
        self.regularization_factor = self.config.get('regularization_factor', 1e-8)
        
        # Monitoring
        self.stabilization_history = []
        self.condition_number_history = []
        
        self.logger.info("Numerical stabilizer initialized")
    
    def stabilize_covariance_update(self, P: np.ndarray, K: np.ndarray, 
                                  H: np.ndarray, R: np.ndarray,
                                  method: str = 'auto') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform numerically stable covariance update
        
        Args:
            P: Prior covariance matrix
            K: Kalman gain matrix
            H: Measurement matrix
            R: Measurement noise covariance
            method: 'joseph', 'standard', or 'auto'
            
        Returns:
            Tuple of (updated_covariance, diagnostics)
        """
        try:
            diagnostics = {
                'method_used': method,
                'condition_number_before': np.linalg.cond(P),
                'stabilization_applied': False
            }
            
            # Choose method automatically if requested
            if method == 'auto':
                method = self._choose_update_method(P, K, H, R)
                diagnostics['method_used'] = method
            
            # Apply chosen method
            if method == 'joseph' or self.use_joseph_form:
                P_updated = self._joseph_form_update(P, K, H, R)
                diagnostics['stabilization_applied'] = True
                
            elif method == 'square_root':
                P_updated = self._square_root_update(P, K, H, R)
                diagnostics['stabilization_applied'] = True
                
            else:
                # Standard update
                I_KH = np.eye(P.shape[0]) - K @ H
                P_updated = I_KH @ P @ I_KH.T + K @ R @ K.T
            
            # Apply regularization if needed
            P_final, reg_applied = self._apply_regularization(P_updated)
            diagnostics['regularization_applied'] = reg_applied
            
            # Final condition number
            diagnostics['condition_number_after'] = np.linalg.cond(P_final)
            
            # Store monitoring data
            self._update_monitoring(diagnostics)
            
            return P_final, diagnostics
            
        except Exception as e:
            self.logger.error(f"Covariance update stabilization failed: {e}")
            # Fallback to regularized prior
            P_regularized, _ = self._apply_regularization(P)
            return P_regularized, {'error': str(e), 'fallback_used': True}
    
    def _choose_update_method(self, P: np.ndarray, K: np.ndarray, 
                            H: np.ndarray, R: np.ndarray) -> str:
        """Automatically choose the best update method"""
        try:
            # Check condition number of prior covariance
            cond_P = np.linalg.cond(P)
            
            # Check if innovation covariance is well-conditioned
            S = H @ P @ H.T + R
            cond_S = np.linalg.cond(S)
            
            # Decision logic
            if cond_P > 1e10 or cond_S > 1e10:
                return 'joseph'  # Use Joseph form for ill-conditioned matrices
            elif self.use_square_root and cond_P > 1e6:
                return 'square_root'  # Use square-root for moderately ill-conditioned
            else:
                return 'standard'  # Use standard update for well-conditioned
                
        except Exception as e:
            self.logger.warning(f"Method selection failed: {e}")
            return 'joseph'  # Default to most stable method
    
    def _joseph_form_update(self, P: np.ndarray, K: np.ndarray, 
                          H: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Joseph form covariance update for guaranteed positive definiteness
        
        P_k+1 = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        """
        try:
            n = P.shape[0]
            I = np.eye(n)
            
            # Compute (I - K*H)
            I_KH = I - K @ H
            
            # Joseph form update
            P_updated = I_KH @ P @ I_KH.T + K @ R @ K.T
            
            # Ensure symmetry
            P_updated = 0.5 * (P_updated + P_updated.T)
            
            return P_updated
            
        except Exception as e:
            self.logger.error(f"Joseph form update failed: {e}")
            raise
    
    def _square_root_update(self, P: np.ndarray, K: np.ndarray, 
                          H: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Square-root covariance update using Cholesky decomposition
        """
        try:
            # Compute square root of P
            try:
                S_P = cholesky(P, lower=True)
            except LinAlgError:
                # If Cholesky fails, use eigenvalue decomposition
                eigenvals, eigenvecs = np.linalg.eigh(P)
                eigenvals = np.maximum(eigenvals, self.min_eigenvalue)
                S_P = eigenvecs @ np.diag(np.sqrt(eigenvals))
            
            # Compute square root of R
            try:
                S_R = cholesky(R, lower=True)
            except LinAlgError:
                eigenvals, eigenvecs = np.linalg.eigh(R)
                eigenvals = np.maximum(eigenvals, self.min_eigenvalue)
                S_R = eigenvecs @ np.diag(np.sqrt(eigenvals))
            
            # Square-root update (simplified implementation)
            # Full implementation would use QR decomposition
            n = P.shape[0]
            I = np.eye(n)
            I_KH = I - K @ H
            
            # Update square root
            S_updated = I_KH @ S_P
            
            # Add measurement noise contribution
            K_SR = K @ S_R
            
            # Combine using QR decomposition (simplified)
            combined = np.hstack([S_updated, K_SR])
            Q, R_qr = np.linalg.qr(combined.T)
            S_final = R_qr[:n, :n].T
            
            # Reconstruct covariance
            P_updated = S_final @ S_final.T
            
            return P_updated
            
        except Exception as e:
            self.logger.warning(f"Square-root update failed: {e}")
            # Fallback to Joseph form
            return self._joseph_form_update(P, K, H, R)
    
    def _apply_regularization(self, P: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Apply adaptive regularization to maintain positive definiteness"""
        try:
            if not self.use_adaptive_regularization:
                return P, False
            
            # Check if regularization is needed
            eigenvals = np.linalg.eigvals(P)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval <= 0 or np.linalg.cond(P) > self.condition_number_threshold:
                # Apply regularization
                reg_amount = max(self.regularization_factor, -min_eigenval + self.min_eigenvalue)
                P_regularized = P + reg_amount * np.eye(P.shape[0])
                
                self.logger.debug(f"Applied regularization: {reg_amount:.2e}")
                return P_regularized, True
            
            return P, False
            
        except Exception as e:
            self.logger.error(f"Regularization failed: {e}")
            # Emergency regularization
            P_emergency = P + self.regularization_factor * np.eye(P.shape[0])
            return P_emergency, True
    
    def stabilize_kalman_gain(self, P: np.ndarray, H: np.ndarray, 
                            R: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute Kalman gain with numerical stabilization
        
        K = P * H^T * (H * P * H^T + R)^(-1)
        """
        try:
            diagnostics = {
                'method_used': 'standard',
                'condition_number_S': None,
                'stabilization_applied': False
            }
            
            # Compute innovation covariance
            S = H @ P @ H.T + R
            diagnostics['condition_number_S'] = np.linalg.cond(S)
            
            # Choose inversion method based on condition number
            if diagnostics['condition_number_S'] > self.condition_number_threshold:
                # Use pseudo-inverse for ill-conditioned matrices
                try:
                    S_inv = np.linalg.pinv(S)
                    diagnostics['method_used'] = 'pseudoinverse'
                    diagnostics['stabilization_applied'] = True
                except Exception:
                    # Fallback to regularized inversion
                    S_reg = S + self.regularization_factor * np.eye(S.shape[0])
                    S_inv = np.linalg.inv(S_reg)
                    diagnostics['method_used'] = 'regularized_inverse'
                    diagnostics['stabilization_applied'] = True
            else:
                # Standard inversion
                try:
                    S_inv = np.linalg.inv(S)
                except LinAlgError:
                    # Fallback to pseudo-inverse
                    S_inv = np.linalg.pinv(S)
                    diagnostics['method_used'] = 'pseudoinverse_fallback'
                    diagnostics['stabilization_applied'] = True
            
            # Compute Kalman gain
            K = P @ H.T @ S_inv
            
            # Check for numerical issues in gain
            if np.any(np.isnan(K)) or np.any(np.isinf(K)):
                self.logger.warning("NaN or Inf detected in Kalman gain")
                # Use conservative gain
                K = 0.1 * P @ H.T / np.trace(S)
                diagnostics['method_used'] = 'conservative_gain'
                diagnostics['stabilization_applied'] = True
            
            return K, diagnostics
            
        except Exception as e:
            self.logger.error(f"Kalman gain computation failed: {e}")
            # Emergency fallback - very conservative gain
            K_emergency = 0.01 * np.eye(P.shape[0], H.shape[0])
            return K_emergency, {'error': str(e), 'emergency_gain': True}
    
    def check_covariance_health(self, P: np.ndarray) -> Dict[str, Any]:
        """Comprehensive covariance matrix health check"""
        try:
            health_report = {
                'is_symmetric': np.allclose(P, P.T),
                'is_positive_definite': None,
                'condition_number': np.linalg.cond(P),
                'min_eigenvalue': None,
                'max_eigenvalue': None,
                'trace': np.trace(P),
                'determinant': np.linalg.det(P),
                'frobenius_norm': np.linalg.norm(P, 'fro'),
                'has_nan': np.any(np.isnan(P)),
                'has_inf': np.any(np.isinf(P)),
                'health_status': 'unknown'
            }
            
            # Eigenvalue analysis
            try:
                eigenvals = np.linalg.eigvals(P)
                health_report['min_eigenvalue'] = np.min(eigenvals)
                health_report['max_eigenvalue'] = np.max(eigenvals)
                health_report['is_positive_definite'] = np.all(eigenvals > 0)
            except Exception as e:
                health_report['eigenvalue_error'] = str(e)
            
            # Overall health assessment
            if health_report['has_nan'] or health_report['has_inf']:
                health_report['health_status'] = 'critical'
            elif not health_report['is_positive_definite']:
                health_report['health_status'] = 'poor'
            elif health_report['condition_number'] > 1e12:
                health_report['health_status'] = 'warning'
            elif not health_report['is_symmetric']:
                health_report['health_status'] = 'warning'
            else:
                health_report['health_status'] = 'good'
            
            return health_report
            
        except Exception as e:
            return {
                'error': str(e),
                'health_status': 'critical'
            }
    
    def repair_covariance(self, P: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Repair a damaged covariance matrix"""
        try:
            repair_log = {
                'original_condition': np.linalg.cond(P),
                'repairs_applied': []
            }
            
            P_repaired = P.copy()
            
            # 1. Handle NaN and Inf values
            if np.any(np.isnan(P_repaired)) or np.any(np.isinf(P_repaired)):
                P_repaired = np.nan_to_num(P_repaired, nan=1.0, posinf=1e6, neginf=-1e6)
                repair_log['repairs_applied'].append('nan_inf_cleanup')
            
            # 2. Enforce symmetry
            if not np.allclose(P_repaired, P_repaired.T):
                P_repaired = 0.5 * (P_repaired + P_repaired.T)
                repair_log['repairs_applied'].append('symmetry_enforcement')
            
            # 3. Ensure positive definiteness
            try:
                eigenvals, eigenvecs = np.linalg.eigh(P_repaired)
                if np.any(eigenvals <= 0):
                    # Clip negative eigenvalues
                    eigenvals = np.maximum(eigenvals, self.min_eigenvalue)
                    P_repaired = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                    repair_log['repairs_applied'].append('positive_definiteness')
            except Exception as e:
                # Emergency repair - add identity
                P_repaired += self.regularization_factor * np.eye(P_repaired.shape[0])
                repair_log['repairs_applied'].append('emergency_regularization')
            
            # 4. Condition number improvement
            if np.linalg.cond(P_repaired) > self.condition_number_threshold:
                # Apply Tikhonov regularization
                reg_strength = np.trace(P_repaired) / P_repaired.shape[0] * 1e-6
                P_repaired += reg_strength * np.eye(P_repaired.shape[0])
                repair_log['repairs_applied'].append('condition_improvement')
            
            repair_log['final_condition'] = np.linalg.cond(P_repaired)
            repair_log['repair_success'] = len(repair_log['repairs_applied']) > 0
            
            return P_repaired, repair_log
            
        except Exception as e:
            self.logger.error(f"Covariance repair failed: {e}")
            # Last resort - identity matrix scaled by trace
            trace_val = np.trace(P) if not np.isnan(np.trace(P)) else 1.0
            P_emergency = (trace_val / P.shape[0]) * np.eye(P.shape[0])
            return P_emergency, {'error': str(e), 'emergency_repair': True}
    
    def _update_monitoring(self, diagnostics: Dict[str, Any]):
        """Update monitoring history"""
        try:
            self.stabilization_history.append({
                'timestamp': len(self.stabilization_history),
                'diagnostics': diagnostics.copy()
            })
            
            if 'condition_number_after' in diagnostics:
                self.condition_number_history.append(diagnostics['condition_number_after'])
            
            # Keep limited history
            if len(self.stabilization_history) > 1000:
                self.stabilization_history = self.stabilization_history[-1000:]
            if len(self.condition_number_history) > 1000:
                self.condition_number_history = self.condition_number_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f"Monitoring update failed: {e}")
    
    def get_stabilization_statistics(self) -> Dict[str, Any]:
        """Get stabilization performance statistics"""
        try:
            if not self.stabilization_history:
                return {'error': 'No stabilization history available'}
            
            # Count stabilization methods used
            methods_used = {}
            stabilizations_applied = 0
            regularizations_applied = 0
            
            for entry in self.stabilization_history:
                diag = entry['diagnostics']
                method = diag.get('method_used', 'unknown')
                methods_used[method] = methods_used.get(method, 0) + 1
                
                if diag.get('stabilization_applied', False):
                    stabilizations_applied += 1
                if diag.get('regularization_applied', False):
                    regularizations_applied += 1
            
            # Condition number statistics
            cond_stats = {}
            if self.condition_number_history:
                cond_numbers = [c for c in self.condition_number_history if not np.isnan(c)]
                if cond_numbers:
                    cond_stats = {
                        'mean': np.mean(cond_numbers),
                        'std': np.std(cond_numbers),
                        'min': np.min(cond_numbers),
                        'max': np.max(cond_numbers),
                        'median': np.median(cond_numbers)
                    }
            
            return {
                'total_updates': len(self.stabilization_history),
                'stabilizations_applied': stabilizations_applied,
                'regularizations_applied': regularizations_applied,
                'stabilization_rate': stabilizations_applied / len(self.stabilization_history),
                'methods_used': methods_used,
                'condition_number_stats': cond_stats
            }
            
        except Exception as e:
            self.logger.error(f"Statistics computation failed: {e}")
            return {'error': str(e)}
    
    def configure_for_accuracy_target(self, target_accuracy_m: float = 500.0):
        """Configure stabilizer for specific accuracy target"""
        try:
            # Adjust thresholds based on accuracy target
            if target_accuracy_m <= 100:  # Very high accuracy
                self.condition_number_threshold = 1e10
                self.regularization_factor = 1e-10
                self.min_eigenvalue = 1e-15
                
            elif target_accuracy_m <= 500:  # Sub-1km accuracy
                self.condition_number_threshold = 1e12
                self.regularization_factor = 1e-8
                self.min_eigenvalue = 1e-12
                
            else:  # Standard accuracy
                self.condition_number_threshold = 1e14
                self.regularization_factor = 1e-6
                self.min_eigenvalue = 1e-10
            
            self.logger.info(f"Configured stabilizer for {target_accuracy_m}m accuracy target")
            
        except Exception as e:
            self.logger.error(f"Configuration failed: {e}")
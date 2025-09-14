import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class MLResidualCorrector:
    """
    Machine Learning residual corrector using LSTM/CNN models
    for systematic bias removal in EKF orbital determination
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ML residual corrector"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_type = config.get('ml_model_type', 'mlp')  # 'mlp', 'rf', 'lstm'
        self.lookback_window = config.get('ml_lookback_window', 10)
        self.feature_dim = config.get('ml_feature_dim', 20)  # Extended feature dimension
        
        # Models
        self.position_model = None
        self.velocity_model = None
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        
        # Training data storage
        self.training_features = []
        self.training_targets = []
        self.is_trained = False
        
        # Model persistence
        self.model_save_path = config.get('ml_model_path', 'models/')
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Performance tracking
        self.training_history = []
        self.prediction_errors = []
        
        self.logger.info(f"ML residual corrector initialized with {self.model_type} model")
    
    def extract_features(self, state: np.ndarray, innovation_history: List[np.ndarray],
                        additional_context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Extract features for ML model from current state and history
        
        Args:
            state: Current state vector [position, velocity, CdA, Cr, empirical_accel]
            innovation_history: Recent innovation vectors
            additional_context: Additional context information
            
        Returns:
            Feature vector for ML model
        """
        try:
            features = []
            
            # Current state features
            position = state[:3]
            velocity = state[3:6]
            
            # Orbital elements derived features
            r_mag = np.linalg.norm(position)
            v_mag = np.linalg.norm(velocity)
            h = np.cross(position, velocity)  # Angular momentum
            h_mag = np.linalg.norm(h)
            
            # Basic orbital parameters
            mu = 3.986004418e14  # Earth gravitational parameter
            energy = v_mag**2 / 2 - mu / r_mag  # Specific energy
            
            if h_mag > 0:
                # Semi-major axis
                a = -mu / (2 * energy) if energy < 0 else r_mag
                
                # Eccentricity vector
                e_vec = ((v_mag**2 - mu/r_mag) * position - np.dot(position, velocity) * velocity) / mu
                eccentricity = np.linalg.norm(e_vec)
                
                # Inclination
                inclination = np.arccos(h[2] / h_mag) if h_mag > 0 else 0
            else:
                a = r_mag
                eccentricity = 0
                inclination = 0
            
            # Add orbital element features
            features.extend([
                r_mag / 6371000,  # Normalized altitude
                v_mag / 7800,     # Normalized velocity
                energy / 1e7,     # Normalized energy
                a / 6371000,      # Normalized semi-major axis
                eccentricity,     # Eccentricity
                inclination,      # Inclination
                h_mag / 1e10      # Normalized angular momentum
            ])
            
            # Current estimated parameters
            if len(state) > 6:
                features.extend([
                    state[6] / 3.0,   # Normalized CdA
                    state[7] / 2.0,   # Normalized Cr
                ])
                if len(state) > 8:
                    features.append(state[8] * 1e6)  # Scaled empirical acceleration
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Innovation history features
            if innovation_history and len(innovation_history) > 0:
                recent_innovations = innovation_history[-min(5, len(innovation_history)):]
                
                # Innovation statistics
                innovation_norms = [np.linalg.norm(innov) for innov in recent_innovations]
                features.extend([
                    np.mean(innovation_norms),
                    np.std(innovation_norms) if len(innovation_norms) > 1 else 0,
                    np.max(innovation_norms),
                    len(recent_innovations)
                ])
                
                # Recent innovation components (position and velocity separately)
                if len(recent_innovations) > 0:
                    last_innovation = recent_innovations[-1]
                    if len(last_innovation) >= 6:
                        pos_innov_norm = np.linalg.norm(last_innovation[:3])
                        vel_innov_norm = np.linalg.norm(last_innovation[3:6])
                        features.extend([pos_innov_norm, vel_innov_norm])
                    else:
                        features.extend([0.0, 0.0])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Additional context features
            if additional_context:
                # Time-based features
                if 'timestamp' in additional_context:
                    timestamp = additional_context['timestamp']
                    if isinstance(timestamp, datetime):
                        # Orbital period approximation
                        orbital_period = 2 * np.pi * np.sqrt(a**3 / mu) if a > 0 else 5400
                        phase = (timestamp.timestamp() % orbital_period) / orbital_period
                        features.extend([
                            np.sin(2 * np.pi * phase),
                            np.cos(2 * np.pi * phase)
                        ])
                    else:
                        features.extend([0.0, 0.0])
                else:
                    features.extend([0.0, 0.0])
                
                # Space weather features
                if 'space_weather' in additional_context:
                    sw = additional_context['space_weather']
                    features.extend([
                        sw.get('f107', 150) / 200,  # Normalized F10.7
                        sw.get('kp', 3) / 9         # Normalized Kp
                    ])
                else:
                    features.extend([0.75, 0.33])  # Default normalized values
            else:
                features.extend([0.0, 0.0, 0.75, 0.33])
            
            # Pad or truncate to desired feature dimension
            while len(features) < self.feature_dim:
                features.append(0.0)
            
            return np.array(features[:self.feature_dim])
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return np.zeros(self.feature_dim)
    
    def add_training_sample(self, state: np.ndarray, innovation_history: List[np.ndarray],
                          true_residual: np.ndarray, additional_context: Optional[Dict[str, Any]] = None):
        """Add a training sample to the dataset"""
        try:
            features = self.extract_features(state, innovation_history, additional_context)
            
            # Split residual into position and velocity components
            if len(true_residual) >= 6:
                pos_residual = true_residual[:3]
                vel_residual = true_residual[3:6]
                
                self.training_features.append(features)
                self.training_targets.append(np.concatenate([pos_residual, vel_residual]))
            
        except Exception as e:
            self.logger.error(f"Training sample addition error: {e}")
    
    def train_models(self, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train ML models on collected training data"""
        try:
            if len(self.training_features) < 10:
                self.logger.warning("Insufficient training data for ML model")
                return {'success': False, 'reason': 'insufficient_data'}
            
            # Convert to arrays
            X = np.array(self.training_features)
            y = np.array(self.training_targets)
            
            # Split into training and validation
            n_samples = len(X)
            n_train = int(n_samples * (1 - validation_split))
            
            X_train, X_val = X[:n_train], X[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]
            
            # Scale features and targets
            X_train_scaled = self.scaler_features.fit_transform(X_train)
            X_val_scaled = self.scaler_features.transform(X_val)
            
            y_train_scaled = self.scaler_targets.fit_transform(y_train)
            y_val_scaled = self.scaler_targets.transform(y_val)
            
            # Train models
            training_results = {}
            
            if self.model_type == 'mlp':
                training_results = self._train_mlp_models(
                    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled
                )
            elif self.model_type == 'rf':
                training_results = self._train_rf_models(
                    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled
                )
            else:
                self.logger.error(f"Unsupported model type: {self.model_type}")
                return {'success': False, 'reason': 'unsupported_model_type'}
            
            if training_results.get('success', False):
                self.is_trained = True
                self.save_models()
                
                # Store training history
                self.training_history.append({
                    'timestamp': datetime.utcnow(),
                    'n_samples': n_samples,
                    'validation_score': training_results.get('validation_score', 0),
                    'training_score': training_results.get('training_score', 0)
                })
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _train_mlp_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train MLP models for position and velocity corrections"""
        try:
            # Position model (first 3 outputs)
            self.position_model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                max_iter=500,
                alpha=0.001,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
            
            # Velocity model (last 3 outputs)
            self.velocity_model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                max_iter=500,
                alpha=0.001,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
            
            # Train position model
            self.position_model.fit(X_train, y_train[:, :3])
            pos_train_score = self.position_model.score(X_train, y_train[:, :3])
            pos_val_score = self.position_model.score(X_val, y_val[:, :3])
            
            # Train velocity model
            self.velocity_model.fit(X_train, y_train[:, 3:6])
            vel_train_score = self.velocity_model.score(X_train, y_train[:, 3:6])
            vel_val_score = self.velocity_model.score(X_val, y_val[:, 3:6])
            
            return {
                'success': True,
                'training_score': (pos_train_score + vel_train_score) / 2,
                'validation_score': (pos_val_score + vel_val_score) / 2,
                'position_train_score': pos_train_score,
                'position_val_score': pos_val_score,
                'velocity_train_score': vel_train_score,
                'velocity_val_score': vel_val_score
            }
            
        except Exception as e:
            self.logger.error(f"MLP training error: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _train_rf_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest models"""
        try:
            # Position model
            self.position_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Velocity model
            self.velocity_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Train models
            self.position_model.fit(X_train, y_train[:, :3])
            self.velocity_model.fit(X_train, y_train[:, 3:6])
            
            # Evaluate
            pos_train_score = self.position_model.score(X_train, y_train[:, :3])
            pos_val_score = self.position_model.score(X_val, y_val[:, :3])
            vel_train_score = self.velocity_model.score(X_train, y_train[:, 3:6])
            vel_val_score = self.velocity_model.score(X_val, y_val[:, 3:6])
            
            return {
                'success': True,
                'training_score': (pos_train_score + vel_train_score) / 2,
                'validation_score': (pos_val_score + vel_val_score) / 2,
                'position_train_score': pos_train_score,
                'position_val_score': pos_val_score,
                'velocity_train_score': vel_train_score,
                'velocity_val_score': vel_val_score
            }
            
        except Exception as e:
            self.logger.error(f"Random Forest training error: {e}")
            return {'success': False, 'reason': str(e)}
    
    def predict_correction(self, state: np.ndarray, innovation_history: List[np.ndarray],
                          additional_context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Predict correction for current state
        
        Args:
            state: Current state vector
            innovation_history: Recent innovation history
            additional_context: Additional context information
            
        Returns:
            Predicted correction for position and velocity [dx, dy, dz, dvx, dvy, dvz]
        """
        try:
            if not self.is_trained or self.position_model is None or self.velocity_model is None:
                return np.zeros(6)
            
            # Extract features
            features = self.extract_features(state, innovation_history, additional_context)
            features_scaled = self.scaler_features.transform(features.reshape(1, -1))
            
            # Predict corrections
            pos_correction_scaled = self.position_model.predict(features_scaled)[0]
            vel_correction_scaled = self.velocity_model.predict(features_scaled)[0]
            
            # Inverse transform
            correction_scaled = np.concatenate([pos_correction_scaled, vel_correction_scaled])
            correction = self.scaler_targets.inverse_transform(correction_scaled.reshape(1, -1))[0]
            
            # Apply magnitude limits for safety
            pos_correction = correction[:3]
            vel_correction = correction[3:6]
            
            # Limit position correction to 1km
            pos_mag = np.linalg.norm(pos_correction)
            if pos_mag > 1000:
                pos_correction = pos_correction * (1000 / pos_mag)
            
            # Limit velocity correction to 10 m/s
            vel_mag = np.linalg.norm(vel_correction)
            if vel_mag > 10:
                vel_correction = vel_correction * (10 / vel_mag)
            
            final_correction = np.concatenate([pos_correction, vel_correction])
            
            # Store prediction for error analysis
            self.prediction_errors.append({
                'timestamp': datetime.utcnow(),
                'correction_magnitude': np.linalg.norm(final_correction),
                'position_correction': np.linalg.norm(pos_correction),
                'velocity_correction': np.linalg.norm(vel_correction)
            })
            
            return final_correction
            
        except Exception as e:
            self.logger.error(f"Correction prediction error: {e}")
            return np.zeros(6)
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            if self.is_trained:
                # Save models
                joblib.dump(self.position_model, 
                          os.path.join(self.model_save_path, 'position_model.pkl'))
                joblib.dump(self.velocity_model, 
                          os.path.join(self.model_save_path, 'velocity_model.pkl'))
                
                # Save scalers
                joblib.dump(self.scaler_features, 
                          os.path.join(self.model_save_path, 'scaler_features.pkl'))
                joblib.dump(self.scaler_targets, 
                          os.path.join(self.model_save_path, 'scaler_targets.pkl'))
                
                # Save training data
                np.save(os.path.join(self.model_save_path, 'training_features.npy'), 
                       self.training_features)
                np.save(os.path.join(self.model_save_path, 'training_targets.npy'), 
                       self.training_targets)
                
                self.logger.info("ML models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            position_model_path = os.path.join(self.model_save_path, 'position_model.pkl')
            velocity_model_path = os.path.join(self.model_save_path, 'velocity_model.pkl')
            
            if os.path.exists(position_model_path) and os.path.exists(velocity_model_path):
                self.position_model = joblib.load(position_model_path)
                self.velocity_model = joblib.load(velocity_model_path)
                
                # Load scalers
                self.scaler_features = joblib.load(
                    os.path.join(self.model_save_path, 'scaler_features.pkl'))
                self.scaler_targets = joblib.load(
                    os.path.join(self.model_save_path, 'scaler_targets.pkl'))
                
                self.is_trained = True
                self.logger.info("ML models loaded successfully")
                return True
            else:
                self.logger.info("No saved models found")
                return False
                
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        try:
            if not self.training_history:
                return {}
            
            latest_training = self.training_history[-1]
            
            performance = {
                'is_trained': self.is_trained,
                'model_type': self.model_type,
                'n_training_samples': len(self.training_features),
                'latest_training_score': latest_training.get('training_score', 0),
                'latest_validation_score': latest_training.get('validation_score', 0),
                'n_predictions': len(self.prediction_errors)
            }
            
            if self.prediction_errors:
                recent_errors = self.prediction_errors[-50:]  # Last 50 predictions
                correction_mags = [err['correction_magnitude'] for err in recent_errors]
                
                performance.update({
                    'recent_correction_mean': np.mean(correction_mags),
                    'recent_correction_std': np.std(correction_mags),
                    'recent_correction_max': np.max(correction_mags)
                })
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Performance analysis error: {e}")
            return {}

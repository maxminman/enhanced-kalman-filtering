import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import os

class SpaceWeatherData:
    """
    Space weather data provider for atmospheric density modeling
    Fetches real-time F10.7 solar flux and Kp geomagnetic indices
    """
    
    def __init__(self):
        """Initialize space weather data provider"""
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        self.noaa_api_base = "https://services.swpc.noaa.gov/json"
        self.celestrak_api = "https://celestrak.org/SpaceData/"
        
        # Current data cache
        self.current_data = {
            'f107': 150.0,  # Default solar flux
            'kp': 3.0,      # Default Kp index
            'ap': 15.0,     # Default Ap index
            'last_update': None,
            'data_age_hours': 999
        }
        
        # Data history for trend analysis
        self.data_history = []
        
        self.logger.info("Space weather data provider initialized")
    
    def update(self) -> bool:
        """
        Update space weather data from external sources
        
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Try multiple data sources
            success = False
            
            # Try NOAA SWPC first
            if self._fetch_noaa_data():
                success = True
                self.logger.info("Space weather data updated from NOAA SWPC")
            
            # Try Celestrak as backup
            elif self._fetch_celestrak_data():
                success = True
                self.logger.info("Space weather data updated from Celestrak")
            
            # Try local fallback data
            elif self._load_fallback_data():
                success = True
                self.logger.info("Using fallback space weather data")
            
            if success:
                self.current_data['last_update'] = datetime.utcnow()
                self.current_data['data_age_hours'] = 0
                
                # Store in history
                self.data_history.append(self.current_data.copy())
                
                # Keep only last 30 days of history
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                self.data_history = [
                    d for d in self.data_history 
                    if d.get('last_update', datetime.min) > cutoff_time
                ]
            
            return success
            
        except Exception as e:
            self.logger.error(f"Space weather update failed: {e}")
            return False
    
    def _fetch_noaa_data(self) -> bool:
        """Fetch data from NOAA Space Weather Prediction Center"""
        try:
            # F10.7 solar flux
            f107_url = f"{self.noaa_api_base}/solar-cycle/observed-solar-cycle-indices.json"
            
            response = requests.get(f107_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Get most recent F10.7 data
                if data and len(data) > 0:
                    latest_entry = data[-1]
                    self.current_data['f107'] = float(latest_entry.get('f10.7', 150.0))
            
            # Kp index
            kp_url = f"{self.noaa_api_base}/planetary_k_index_1m.json"
            
            response = requests.get(kp_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Get most recent Kp data
                if data and len(data) > 0:
                    latest_entry = data[-1]
                    kp_value = latest_entry.get('kp', 3.0)
                    
                    # Convert to numeric if needed
                    if isinstance(kp_value, str):
                        # Handle string formats like "3o", "4-", etc.
                        kp_numeric = float(kp_value.replace('o', '.33').replace('-', '.00').replace('+', '.67')[:3])
                        self.current_data['kp'] = kp_numeric
                    else:
                        self.current_data['kp'] = float(kp_value)
                    
                    # Calculate Ap from Kp
                    self.current_data['ap'] = self._kp_to_ap(self.current_data['kp'])
            
            return True
            
        except Exception as e:
            self.logger.warning(f"NOAA data fetch failed: {e}")
            return False
    
    def _fetch_celestrak_data(self) -> bool:
        """Fetch data from Celestrak space weather service"""
        try:
            # Celestrak space weather data
            sw_url = f"{self.celestrak_api}SW-All.txt"
            
            response = requests.get(sw_url, timeout=10)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                
                # Parse the most recent data line
                for line in reversed(lines):
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 10:
                            try:
                                # Celestrak format: YYYY MM DD ... F10.7 ... Kp ...
                                f107_index = 6  # Approximate index for F10.7
                                kp_index = 8    # Approximate index for Kp
                                
                                if len(parts) > f107_index:
                                    self.current_data['f107'] = float(parts[f107_index])
                                
                                if len(parts) > kp_index:
                                    self.current_data['kp'] = float(parts[kp_index])
                                    self.current_data['ap'] = self._kp_to_ap(self.current_data['kp'])
                                
                                return True
                                
                            except (ValueError, IndexError):
                                continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Celestrak data fetch failed: {e}")
            return False
    
    def _load_fallback_data(self) -> bool:
        """Load fallback space weather data from local source"""
        try:
            # Use historical average values based on solar cycle
            current_year = datetime.utcnow().year
            
            # Estimate solar cycle phase (simplified 11-year cycle)
            solar_cycle_year = (current_year - 2008) % 11  # 2008 was near solar minimum
            
            if solar_cycle_year < 4:  # Solar minimum to rising
                self.current_data['f107'] = 80 + 20 * solar_cycle_year
                self.current_data['kp'] = 1.0 + 0.5 * solar_cycle_year
            elif solar_cycle_year < 7:  # Solar maximum
                self.current_data['f107'] = 160 + 40 * np.sin(np.pi * (solar_cycle_year - 4) / 3)
                self.current_data['kp'] = 3.0 + 2.0 * np.sin(np.pi * (solar_cycle_year - 4) / 3)
            else:  # Solar maximum to minimum
                phase = (solar_cycle_year - 7) / 4
                self.current_data['f107'] = 160 * (1 - 0.6 * phase)
                self.current_data['kp'] = 5.0 * (1 - 0.8 * phase)
            
            # Add some random variation
            self.current_data['f107'] += np.random.normal(0, 10)
            self.current_data['kp'] += np.random.normal(0, 0.5)
            
            # Ensure reasonable bounds
            self.current_data['f107'] = max(65, min(300, self.current_data['f107']))
            self.current_data['kp'] = max(0, min(9, self.current_data['kp']))
            
            self.current_data['ap'] = self._kp_to_ap(self.current_data['kp'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fallback data loading failed: {e}")
            return False
    
    def _kp_to_ap(self, kp: float) -> float:
        """Convert Kp index to Ap index"""
        # Standard Kp to Ap conversion table
        kp_ap_table = {
            0.0: 0, 0.33: 2, 0.67: 3, 1.0: 4, 1.33: 5, 1.67: 6, 2.0: 7,
            2.33: 9, 2.67: 12, 3.0: 15, 3.33: 18, 3.67: 22, 4.0: 27,
            4.33: 32, 4.67: 39, 5.0: 48, 5.33: 56, 5.67: 67, 6.0: 80,
            6.33: 94, 6.67: 111, 7.0: 132, 7.33: 154, 7.67: 179, 8.0: 207,
            8.33: 236, 8.67: 300, 9.0: 400
        }
        
        # Find closest Kp value in table
        kp_values = sorted(kp_ap_table.keys())
        closest_kp = min(kp_values, key=lambda x: abs(x - kp))
        
        return kp_ap_table[closest_kp]
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current space weather data"""
        # Update data age
        if self.current_data['last_update']:
            age = datetime.utcnow() - self.current_data['last_update']
            self.current_data['data_age_hours'] = age.total_seconds() / 3600
        
        return self.current_data.copy()
    
    def get_historical_data(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get historical space weather data"""
        cutoff_time = datetime.utcnow() - timedelta(days=days_back)
        
        return [
            d for d in self.data_history
            if d.get('last_update', datetime.min) > cutoff_time
        ]
    
    def get_f107_average(self, days: int = 81) -> float:
        """Get F10.7 average over specified days (default 81-day average)"""
        recent_data = self.get_historical_data(days)
        
        if len(recent_data) > 0:
            f107_values = [d['f107'] for d in recent_data if 'f107' in d]
            if f107_values:
                return np.mean(f107_values)
        
        # Return current value if no history
        return self.current_data['f107']
    
    def is_data_fresh(self, max_age_hours: int = 24) -> bool:
        """Check if current data is fresh"""
        return self.current_data['data_age_hours'] < max_age_hours
    
    def get_space_weather_status(self) -> str:
        """Get qualitative space weather status"""
        kp = self.current_data['kp']
        f107 = self.current_data['f107']
        
        if kp < 4 and f107 < 120:
            return "Quiet"
        elif kp < 5 and f107 < 150:
            return "Unsettled"
        elif kp < 6 and f107 < 200:
            return "Active"
        elif kp < 7:
            return "Minor Storm"
        elif kp < 8:
            return "Moderate Storm"
        else:
            return "Major Storm"

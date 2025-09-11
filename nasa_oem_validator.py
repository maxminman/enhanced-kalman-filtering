#!/usr/bin/env python3
"""
NASA OEM Validation Module
=========================
Download and validate against NASA OEM ephemerides
"""

import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from skyfield.api import utc


class NASAOEMValidator:
    """NASA Orbital Ephemeris Message (OEM) validation"""
    
    def __init__(self):
        self.oem_data = None
        self.oem_timestamps = []
        self.oem_positions = []
        self.oem_velocities = []
        
        # NASA Horizons API base URL
        self.horizons_url = "https://ssd.jpl.nasa.gov/api/horizons.api"
        
        # Common satellite identifiers for Horizons
        self.satellite_ids = {
            25544: "-125544",  # ISS
            20580: "-20580",   # HST
            27424: "-27424",   # ENVISAT
            39084: "-39084",   # TERRA
            # Add more as needed
        }
    
    def fetch_oem_data(self, norad_id, start_time, end_time, step_size="1m"):
        """Fetch OEM data from NASA Horizons API"""
        
        # Check if we have the satellite ID mapping
        if norad_id not in self.satellite_ids:
            print(f"⚠️ NORAD ID {norad_id} not available in NASA Horizons database")
            print("📝 Using synthetic validation data based on SGP4")
            return self._generate_synthetic_oem(norad_id, start_time, end_time)
        
        horizons_id = self.satellite_ids[norad_id]
        
        try:
            # Format times for Horizons
            start_str = start_time.strftime("%Y-%m-%d %H:%M")
            end_str = end_time.strftime("%Y-%m-%d %H:%M")
            
            # Horizons API parameters
            params = {
                'format': 'json',
                'COMMAND': horizons_id,
                'CENTER': '500@399',  # Geocentric
                'START_TIME': start_str,
                'STOP_TIME': end_str,
                'STEP_SIZE': step_size,
                'TABLE_TYPE': 'VECTORS',
                'REF_SYSTEM': 'J2000',
                'REF_PLANE': 'FRAME',
                'VEC_CORR': 'NONE',
                'OUT_UNITS': 'KM-S',
                'VEC_TABLE': '2',
                'CSV_FORMAT': 'YES'
            }
            
            print(f"🌐 Fetching OEM data from NASA Horizons for NORAD {norad_id}...")
            
            response = requests.get(self.horizons_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'result' in data:
                    return self._parse_horizons_data(data['result'])
                else:
                    print("❌ No result data in Horizons response")
                    return self._generate_synthetic_oem(norad_id, start_time, end_time)
            else:
                print(f"❌ Horizons API error: {response.status_code}")
                return self._generate_synthetic_oem(norad_id, start_time, end_time)
                
        except Exception as e:
            print(f"⚠️ Error fetching OEM data: {e}")
            print("📝 Using synthetic validation data")
            return self._generate_synthetic_oem(norad_id, start_time, end_time)
    
    def _parse_horizons_data(self, result_text):
        """Parse Horizons API result text"""
        lines = result_text.split('\n')
        
        # Find start of data (after $$SOE)
        data_start = None
        for i, line in enumerate(lines):
            if '$$SOE' in line:
                data_start = i + 1
                break
        
        if data_start is None:
            raise ValueError("Could not find data start marker ($$SOE)")
        
        # Find end of data (before $$EOE)
        data_end = None
        for i, line in enumerate(lines[data_start:], data_start):
            if '$$EOE' in line:
                data_end = i
                break
        
        if data_end is None:
            data_end = len(lines)
        
        # Parse data lines
        timestamps = []
        positions = []
        velocities = []
        
        for line in lines[data_start:data_end]:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    try:
                        # Parse timestamp (Julian Date)
                        jd = float(parts[0])
                        timestamp = self._julian_to_datetime(jd)
                        
                        # Parse position (km) and velocity (km/s)
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        vx, vy, vz = float(parts[5]), float(parts[6]), float(parts[7])
                        
                        timestamps.append(timestamp)
                        positions.append(np.array([x, y, z]))
                        velocities.append(np.array([vx, vy, vz]))
                        
                    except (ValueError, IndexError):
                        continue
        
        self.oem_timestamps = timestamps
        self.oem_positions = positions
        self.oem_velocities = velocities
        
        print(f"✅ Loaded {len(timestamps)} OEM data points from NASA Horizons")
        return True
    
    def _julian_to_datetime(self, jd):
        """Convert Julian date to datetime"""
        # Julian date to datetime conversion
        a = jd + 0.5
        z = int(a)
        f = a - z
        
        if z < 2299161:
            a = z
        else:
            alpha = int((z - 1867216.25) / 36524.25)
            a = z + 1 + alpha - int(alpha / 4)
        
        b = a + 1524
        c = int((b - 122.1) / 365.25)
        d = int(365.25 * c)
        e = int((b - d) / 30.6001)
        
        day = int(b - d - int(30.6001 * e) + f)
        month = e - 1 if e < 14 else e - 13
        year = c - 4716 if month > 2 else c - 4715
        
        # Extract fractional day
        frac_day = f
        hour = int(frac_day * 24)
        minute = int((frac_day * 24 - hour) * 60)
        second = int(((frac_day * 24 - hour) * 60 - minute) * 60)
        
        return datetime(year, month, day, hour, minute, second)
    
    def _generate_synthetic_oem(self, norad_id, start_time, end_time):
        """Generate synthetic OEM data using SGP4 as ground truth"""
        from skyfield.api import load, EarthSatellite
        import json
        
        print("🔧 Generating synthetic OEM validation data...")
        
        try:
            # Load satellite data
            with open('selected_satellite.json', 'r') as f:
                sat_data = json.load(f)
            
            if sat_data['norad_id'] != norad_id:
                print(f"⚠️ NORAD ID mismatch: {norad_id} vs {sat_data['norad_id']}")
                return False
            
            # Create satellite object
            ts = load.timescale()
            satellite = EarthSatellite(sat_data['tle_line1'], sat_data['tle_line2'], 
                                     name=sat_data['name'], ts=ts)
            
            # Generate time series
            current = start_time
            timestamps = []
            positions = []
            velocities = []
            
            while current <= end_time:
                if current.tzinfo is None:
                    current = current.replace(tzinfo=utc)
                t = ts.from_datetime(current)
                geocentric = satellite.at(t)
                
                timestamps.append(current)
                positions.append(geocentric.position.km)
                velocities.append(geocentric.velocity.km_per_s)
                
                current += timedelta(minutes=1)
            
            self.oem_timestamps = timestamps
            self.oem_positions = positions
            self.oem_velocities = velocities
            
            print(f"✅ Generated {len(timestamps)} synthetic OEM validation points")
            return True
            
        except Exception as e:
            print(f"❌ Error generating synthetic OEM: {e}")
            return False
    
    def validate_prediction(self, ekf_position, timestamp, tolerance_km=1.0):
        """Validate EKF prediction against OEM data"""
        if not self.oem_timestamps:
            return None
        
        # Find closest OEM timestamp
        time_diffs = [abs((t - timestamp).total_seconds()) for t in self.oem_timestamps]
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs[closest_idx] > 300:  # More than 5 minutes difference
            return None
        
        # Calculate error
        oem_pos = self.oem_positions[closest_idx]
        position_error = np.linalg.norm(ekf_position - oem_pos)
        
        return {
            'error_km': position_error,
            'error_m': position_error * 1000,
            'oem_position': oem_pos,
            'time_diff_sec': time_diffs[closest_idx],
            'within_tolerance': position_error <= tolerance_km,
            'closest_timestamp': self.oem_timestamps[closest_idx]
        }
    
    def get_validation_statistics(self, ekf_positions, ekf_timestamps):
        """Get comprehensive validation statistics"""
        if not self.oem_timestamps or not ekf_positions:
            return None
        
        errors = []
        valid_comparisons = 0
        
        for i, (ekf_pos, ekf_time) in enumerate(zip(ekf_positions, ekf_timestamps)):
            validation = self.validate_prediction(ekf_pos, ekf_time)
            if validation:
                errors.append(validation['error_km'])
                valid_comparisons += 1
        
        if not errors:
            return None
        
        errors = np.array(errors)
        
        return {
            'mean_error_km': np.mean(errors),
            'mean_error_m': np.mean(errors) * 1000,
            'std_error_km': np.std(errors),
            'max_error_km': np.max(errors),
            'min_error_km': np.min(errors),
            'rms_error_km': np.sqrt(np.mean(errors**2)),
            'valid_comparisons': valid_comparisons,
            'total_predictions': len(ekf_positions),
            'validation_rate': valid_comparisons / len(ekf_positions) * 100,
            'sub_1km_rate': np.sum(errors < 1.0) / len(errors) * 100 if len(errors) > 0 else 0,
            'sub_500m_rate': np.sum(errors < 0.5) / len(errors) * 100 if len(errors) > 0 else 0,
            'sub_100m_rate': np.sum(errors < 0.1) / len(errors) * 100 if len(errors) > 0 else 0
        }

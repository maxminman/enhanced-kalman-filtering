#!/usr/bin/env python3
"""
Satellite Data Management Module
===============================
Handle satellite data loading and TLE management
"""

import json
import requests
import numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite
import warnings
warnings.filterwarnings('ignore')


class SatelliteDataManager:
    """Manage satellite data and TLE information"""
    
    def __init__(self):
        self.ts = load.timescale()
        self.satellite = None
        self.satellite_data = None
        
        # TLE sources
        self.tle_sources = {
            'celestrak_stations': 'https://celestrak.org/NORAD/elements/stations.txt',
            'celestrak_visual': 'https://celestrak.org/NORAD/elements/visual.txt',
            'celestrak_active': 'https://celestrak.org/NORAD/elements/active.txt'
        }
    
    def load_selected_satellite(self, filename='selected_satellite.json'):
        """Load satellite from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.satellite_data = json.load(f)
            
            name = self.satellite_data['name']
            line1 = self.satellite_data['tle_line1']
            line2 = self.satellite_data['tle_line2']
            
            self.satellite = EarthSatellite(line1, line2, name=name, ts=self.ts)
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: {filename} not found")
            return False
        except Exception as e:
            print(f"❌ Error loading satellite: {e}")
            return False
    
    def update_tle_from_celestrak(self, norad_id):
        """Update TLE from CelesTrak"""
        try:
            for source_name, url in self.tle_sources.items():
                print(f"🌐 Checking {source_name}...")
                
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                
                lines = response.text.strip().split('\n')
                
                # Parse TLE data
                for i in range(0, len(lines), 3):
                    if i + 2 >= len(lines):
                        break
                    
                    name = lines[i].strip()
                    line1 = lines[i + 1].strip()
                    line2 = lines[i + 2].strip()
                    
                    # Check if this is our satellite
                    if line1.startswith('1') and line2.startswith('2'):
                        try:
                            catalog_num = int(line1[2:7])
                            if catalog_num == norad_id:
                                print(f"✅ Found updated TLE for {name}")
                                return {
                                    'name': name,
                                    'tle_line1': line1,
                                    'tle_line2': line2
                                }
                        except ValueError:
                            continue
            
            print(f"⚠️ No TLE found for NORAD ID {norad_id}")
            return None
            
        except Exception as e:
            print(f"❌ Error updating TLE: {e}")
            return None
    
    def get_sgp4_state_at_time(self, timestamp):
        """Get SGP4 state vector at specific time"""
        if not self.satellite:
            return None
        
        # Convert to Skyfield time
        if isinstance(timestamp, datetime):
            t = self.ts.from_datetime(timestamp.replace(tzinfo=None))
        else:
            t = timestamp
        
        # Get geocentric state
        geocentric = self.satellite.at(t)
        
        return {
            'position_km': geocentric.position.km,
            'velocity_km_s': geocentric.velocity.km_per_s,
            'timestamp': timestamp if isinstance(timestamp, datetime) else t.utc_datetime()
        }
    
    def get_orbital_elements(self, timestamp=None):
        """Extract orbital elements from TLE"""
        if not self.satellite:
            return None
        
        # Parse TLE line 2 for orbital elements
        line2 = self.satellite_data['tle_line2']
        
        try:
            # Inclination (degrees)
            inclination = float(line2[8:16])
            
            # Right Ascension of Ascending Node (degrees)
            raan = float(line2[17:25])
            
            # Eccentricity
            ecc_str = '0.' + line2[26:33]
            eccentricity = float(ecc_str)
            
            # Argument of Perigee (degrees)
            arg_perigee = float(line2[34:42])
            
            # Mean Anomaly (degrees)
            mean_anomaly = float(line2[43:51])
            
            # Mean Motion (revolutions per day)
            mean_motion = float(line2[52:63])
            
            # Calculate semi-major axis from mean motion
            # n = sqrt(mu/a^3), so a = (mu/n^2)^(1/3)
            n_rad_per_sec = mean_motion * 2 * np.pi / 86400  # Convert to rad/s
            mu = 398600.4418  # km^3/s^2
            semi_major_axis = (mu / (n_rad_per_sec**2))**(1/3)
            
            return {
                'semi_major_axis_km': semi_major_axis,
                'eccentricity': eccentricity,
                'inclination_deg': inclination,
                'raan_deg': raan,
                'arg_perigee_deg': arg_perigee,
                'mean_anomaly_deg': mean_anomaly,
                'mean_motion_rev_per_day': mean_motion,
                'period_minutes': 1440 / mean_motion  # minutes per orbit
            }
            
        except (ValueError, IndexError) as e:
            print(f"❌ Error parsing orbital elements: {e}")
            return None
    
    def estimate_measurement_noise(self):
        """Estimate SGP4 measurement noise characteristics"""
        if not self.satellite_data:
            return None
        
        # Base noise estimates for SGP4
        tle_age_hours = self.satellite_data.get('tle_age_hours', 24)
        altitude_km = self.satellite_data.get('altitude_km', 400)
        
        # Position noise increases with TLE age and altitude
        base_pos_noise_km = 0.1  # 100m base uncertainty
        age_factor = 1 + (tle_age_hours / 24) * 0.5  # 50% increase per day
        alt_factor = 1 + max(0, (altitude_km - 300) / 1000) * 0.2  # 20% per 1000km above 300km
        
        position_noise_km = base_pos_noise_km * age_factor * alt_factor
        
        # Velocity noise is typically 1% of position noise
        velocity_noise_km_s = position_noise_km * 0.01
        
        return {
            'position_noise_km': position_noise_km,
            'velocity_noise_km_s': velocity_noise_km_s,
            'tle_age_hours': tle_age_hours,
            'altitude_km': altitude_km
        }
    
    def predict_ground_track(self, start_time, duration_hours=2, step_minutes=1):
        """Predict ground track for visualization"""
        if not self.satellite:
            return []
        
        ground_track = []
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        while current_time <= end_time:
            state = self.get_sgp4_state_at_time(current_time)
            if state:
                # Convert to lat/lon using Skyfield
                t = self.ts.from_datetime(current_time.replace(tzinfo=None))
                geocentric = self.satellite.at(t)
                subpoint = geocentric.subpoint()
                
                ground_track.append({
                    'timestamp': current_time,
                    'latitude': subpoint.latitude.degrees,
                    'longitude': subpoint.longitude.degrees,
                    'altitude': subpoint.elevation.km
                })
            
            current_time += timedelta(minutes=step_minutes)
        
        return ground_track

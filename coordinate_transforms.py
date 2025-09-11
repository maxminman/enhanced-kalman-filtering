#!/usr/bin/env python3
"""
Coordinate Transformation Module
===============================
Proper ECI ↔ ECEF ↔ Geodetic transformations
"""

import numpy as np
from datetime import datetime, timedelta
import math


class CoordinateTransforms:
    """Coordinate transformation utilities"""
    
    def __init__(self):
        # WGS84 parameters
        self.a = 6378137.0  # Semi-major axis (meters)
        self.f = 1/298.257223563  # Flattening
        self.e2 = 2*self.f - self.f**2  # First eccentricity squared
        self.omega_e = 7.2921159e-5  # Earth rotation rate (rad/s)
        
        # Convert to km for consistency
        self.a_km = self.a / 1000.0
    
    def julian_date(self, dt):
        """Convert datetime to Julian date"""
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12*a - 3
        
        jdn = dt.day + (153*m + 2) // 5 + 365*y + y//4 - y//100 + y//400 - 32045
        
        # Add fractional day
        frac_day = (dt.hour + dt.minute/60.0 + dt.second/3600.0) / 24.0
        
        return jdn + frac_day - 0.5
    
    def greenwich_mean_sidereal_time(self, jd):
        """Calculate Greenwich Mean Sidereal Time"""
        # Days since J2000.0
        t = (jd - 2451545.0) / 36525.0
        
        # GMST in seconds
        gmst_sec = 67310.54841 + (876600*3600 + 8640184.812866)*t + 0.093104*t**2 - 6.2e-6*t**3
        
        # Convert to radians and normalize
        gmst_rad = (gmst_sec % 86400) * (2 * np.pi / 86400)
        
        return gmst_rad
    
    def eci_to_ecef(self, eci_pos, eci_vel, timestamp):
        """Convert ECI to ECEF coordinates"""
        # Calculate Greenwich Mean Sidereal Time
        jd = self.julian_date(timestamp)
        gmst = self.greenwich_mean_sidereal_time(jd)
        
        # Rotation matrix from ECI to ECEF
        cos_gmst = np.cos(gmst)
        sin_gmst = np.sin(gmst)
        
        R = np.array([
            [cos_gmst, sin_gmst, 0],
            [-sin_gmst, cos_gmst, 0],
            [0, 0, 1]
        ])
        
        # Transform position
        ecef_pos = R @ eci_pos
        
        # Transform velocity (account for Earth rotation)
        omega_cross_r = np.array([-self.omega_e * eci_pos[1], 
                                  self.omega_e * eci_pos[0], 
                                  0])
        ecef_vel = R @ (eci_vel - omega_cross_r)
        
        return ecef_pos, ecef_vel
    
    def ecef_to_eci(self, ecef_pos, ecef_vel, timestamp):
        """Convert ECEF to ECI coordinates"""
        # Calculate Greenwich Mean Sidereal Time
        jd = self.julian_date(timestamp)
        gmst = self.greenwich_mean_sidereal_time(jd)
        
        # Rotation matrix from ECEF to ECI (transpose of ECI to ECEF)
        cos_gmst = np.cos(gmst)
        sin_gmst = np.sin(gmst)
        
        R = np.array([
            [cos_gmst, -sin_gmst, 0],
            [sin_gmst, cos_gmst, 0],
            [0, 0, 1]
        ])
        
        # Transform position
        eci_pos = R @ ecef_pos
        
        # Transform velocity (account for Earth rotation)
        eci_vel = R @ ecef_vel
        omega_cross_r = np.array([-self.omega_e * eci_pos[1], 
                                  self.omega_e * eci_pos[0], 
                                  0])
        eci_vel += omega_cross_r
        
        return eci_pos, eci_vel
    
    def ecef_to_geodetic(self, ecef_pos_km):
        """Convert ECEF position to geodetic coordinates (lat, lon, alt)"""
        # Convert km to meters for calculation
        x, y, z = ecef_pos_km * 1000
        
        # Longitude
        lon = np.arctan2(y, x)
        
        # Latitude and altitude using iterative method
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p * (1 - self.e2))
        
        # Iterate to improve accuracy
        for _ in range(5):
            N = self.a / np.sqrt(1 - self.e2 * np.sin(lat)**2)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - self.e2 * N / (N + alt)))
        
        # Final altitude calculation
        N = self.a / np.sqrt(1 - self.e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        
        return {
            'latitude': np.degrees(lat),
            'longitude': np.degrees(lon),
            'altitude': alt / 1000.0  # Convert back to km
        }
    
    def geodetic_to_ecef(self, lat_deg, lon_deg, alt_km):
        """Convert geodetic coordinates to ECEF position"""
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        alt_m = alt_km * 1000  # Convert to meters
        
        # Prime vertical radius of curvature
        N = self.a / np.sqrt(1 - self.e2 * np.sin(lat)**2)
        
        # ECEF coordinates in meters
        x = (N + alt_m) * np.cos(lat) * np.cos(lon)
        y = (N + alt_m) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - self.e2) + alt_m) * np.sin(lat)
        
        # Convert to km
        return np.array([x, y, z]) / 1000.0
    
    def eci_to_geodetic(self, eci_pos, eci_vel, timestamp):
        """Direct ECI to geodetic conversion"""
        # Convert ECI to ECEF
        ecef_pos, ecef_vel = self.eci_to_ecef(eci_pos, eci_vel, timestamp)
        
        # Convert ECEF to geodetic
        geodetic = self.ecef_to_geodetic(ecef_pos)
        
        return geodetic
    
    def geodetic_to_eci(self, lat_deg, lon_deg, alt_km, timestamp):
        """Convert geodetic coordinates to ECI"""
        # Convert geodetic to ECEF
        ecef_pos = self.geodetic_to_ecef(lat_deg, lon_deg, alt_km)
        
        # Assume zero velocity in ECEF frame for position-only conversion
        ecef_vel = np.zeros(3)
        
        # Convert ECEF to ECI
        eci_pos, eci_vel = self.ecef_to_eci(ecef_pos, ecef_vel, timestamp)
        
        return eci_pos, eci_vel

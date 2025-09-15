import numpy as np
from typing import Tuple, Optional
from datetime import datetime
import math
try:
    from skyfield.api import load, utc
    from skyfield.earthlib import earth_rotation_angle
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

class CoordinateTransforms:
    """
    Coordinate transformation utilities for orbital mechanics
    Supporting ECI, ECEF, and geodetic coordinate systems
    """
    
    def __init__(self):
        """Initialize coordinate transformation constants"""
        # WGS84 ellipsoid parameters
        self.a = 6378137.0          # Semi-major axis (m)
        self.f = 1 / 298.257223563  # Flattening
        self.b = self.a * (1 - self.f)  # Semi-minor axis (m)
        self.e2 = 2 * self.f - self.f**2  # First eccentricity squared
        
        # Earth rotation rate
        self.omega_earth = 7.2921159e-5  # rad/s
        
        # Earth orientation parameters (simplified - real implementation would use IERS data)
        self.x_pole = 0.0  # arcsec
        self.y_pole = 0.0  # arcsec  
        self.dut1 = 0.0    # seconds
        
        # Initialize time scale if available
        if SKYFIELD_AVAILABLE:
            try:
                self.ts = load.timescale()
            except:
                self.ts = None
        else:
            self.ts = None
        
        # Gravitational parameter
        self.mu = 3.986004418e14  # m^3/s^2
        
        # Constants for improved time and coordinate transformations
        self.j2000_epoch = 2451545.0  # Julian date of J2000.0
        self.seconds_per_day = 86400.0
        self.arcsec_to_rad = 4.84813681109536e-6  # Convert arcseconds to radians
    
    def eci_to_ecef(self, position_eci: np.ndarray, velocity_eci: np.ndarray, 
                   datetime_utc: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert ECI coordinates to ECEF
        
        Args:
            position_eci: Position vector in ECI frame (m)
            velocity_eci: Velocity vector in ECI frame (m/s)
            datetime_utc: UTC datetime
            
        Returns:
            Tuple of (position_ecef, velocity_ecef)
        """
        # Greenwich Mean Sidereal Time
        gmst = self._greenwich_mean_sidereal_time(datetime_utc)
        
        # Rotation matrix from ECI to ECEF
        cos_gmst = np.cos(gmst)
        sin_gmst = np.sin(gmst)
        
        R_eci_to_ecef = np.array([
            [cos_gmst, sin_gmst, 0],
            [-sin_gmst, cos_gmst, 0],
            [0, 0, 1]
        ])
        
        # Transform position
        position_ecef = R_eci_to_ecef @ position_eci
        
        # Transform velocity (includes Earth rotation effect)
        omega_cross_r = np.array([
            -self.omega_earth * position_ecef[1],
            self.omega_earth * position_ecef[0],
            0
        ])
        
        velocity_ecef = R_eci_to_ecef @ velocity_eci - omega_cross_r
        
        return position_ecef, velocity_ecef
    
    def ecef_to_eci(self, position_ecef: np.ndarray, velocity_ecef: np.ndarray,
                   datetime_utc: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert ECEF coordinates to ECI
        
        Args:
            position_ecef: Position vector in ECEF frame (m)
            velocity_ecef: Velocity vector in ECEF frame (m/s)
            datetime_utc: UTC datetime
            
        Returns:
            Tuple of (position_eci, velocity_eci)
        """
        # Greenwich Mean Sidereal Time
        gmst = self._greenwich_mean_sidereal_time(datetime_utc)
        
        # Rotation matrix from ECEF to ECI
        cos_gmst = np.cos(gmst)
        sin_gmst = np.sin(gmst)
        
        R_ecef_to_eci = np.array([
            [cos_gmst, -sin_gmst, 0],
            [sin_gmst, cos_gmst, 0],
            [0, 0, 1]
        ])
        
        # Transform position
        position_eci = R_ecef_to_eci @ position_ecef
        
        # Transform velocity (includes Earth rotation effect)
        omega_cross_r = np.array([
            -self.omega_earth * position_ecef[1],
            self.omega_earth * position_ecef[0],
            0
        ])
        
        velocity_eci = R_ecef_to_eci @ (velocity_ecef + omega_cross_r)
        
        return position_eci, velocity_eci
    
    def ecef_to_geodetic(self, position_ecef: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert ECEF coordinates to geodetic (latitude, longitude, altitude)
        
        Args:
            position_ecef: Position vector in ECEF frame (m)
            
        Returns:
            Tuple of (latitude_deg, longitude_deg, altitude_m)
        """
        x, y, z = position_ecef
        
        # Longitude
        longitude = np.arctan2(y, x)
        
        # Iterative solution for latitude and altitude
        p = np.sqrt(x**2 + y**2)  # Distance from z-axis
        
        # Initial guess
        latitude = np.arctan2(z, p * (1 - self.e2))
        altitude = 0
        
        # Iterative refinement
        for _ in range(10):
            N = self.a / np.sqrt(1 - self.e2 * np.sin(latitude)**2)
            altitude = p / np.cos(latitude) - N
            latitude = np.arctan2(z, p * (1 - self.e2 * N / (N + altitude)))
        
        # Convert to degrees
        latitude_deg = np.degrees(latitude)
        longitude_deg = np.degrees(longitude)
        
        return latitude_deg, longitude_deg, altitude
    
    def geodetic_to_ecef(self, latitude_deg: float, longitude_deg: float, 
                        altitude_m: float) -> np.ndarray:
        """
        Convert geodetic coordinates to ECEF
        
        Args:
            latitude_deg: Latitude in degrees
            longitude_deg: Longitude in degrees
            altitude_m: Altitude in meters
            
        Returns:
            Position vector in ECEF frame (m)
        """
        lat_rad = np.radians(latitude_deg)
        lon_rad = np.radians(longitude_deg)
        
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)
        
        # Radius of curvature in prime vertical
        N = self.a / np.sqrt(1 - self.e2 * sin_lat**2)
        
        # ECEF coordinates
        x = (N + altitude_m) * cos_lat * cos_lon
        y = (N + altitude_m) * cos_lat * sin_lon
        z = (N * (1 - self.e2) + altitude_m) * sin_lat
        
        return np.array([x, y, z])
    
    def eci_to_geodetic(self, position_eci: np.ndarray, 
                       datetime_utc: datetime) -> Tuple[float, float, float]:
        """
        Convert ECI coordinates directly to geodetic
        
        Args:
            position_eci: Position vector in ECI frame (m)
            datetime_utc: UTC datetime
            
        Returns:
            Tuple of (latitude_deg, longitude_deg, altitude_m)
        """
        # Convert ECI to ECEF first
        velocity_dummy = np.zeros(3)
        position_ecef, _ = self.eci_to_ecef(position_eci, velocity_dummy, datetime_utc)
        
        # Convert ECEF to geodetic
        return self.ecef_to_geodetic(position_ecef)
    
    def _greenwich_mean_sidereal_time(self, datetime_utc: datetime) -> float:
        """
        Calculate enhanced Greenwich Mean Sidereal Time with higher accuracy
        
        Args:
            datetime_utc: UTC datetime
            
        Returns:
            GMST in radians
        """
        if SKYFIELD_AVAILABLE and self.ts is not None:
            # Use Skyfield for highest accuracy
            try:
                t = self.ts.from_datetime(datetime_utc.replace(tzinfo=utc))
                # Get Earth rotation angle (more accurate than GMST for modern applications)
                era = earth_rotation_angle(t.tt)
                return era
            except:
                pass  # Fall back to analytical calculation
        
        # Analytical GMST calculation (IAU 2000 model)
        jd = self._julian_date(datetime_utc)
        
        # Modified Julian Date
        mjd = jd - 2400000.5
        
        # Centuries since J2000.0
        T = (jd - self.j2000_epoch) / 36525.0
        
        # GMST at 0h UT (IAU 2000 model)
        gmst0 = (24110.54841 + 
                8640184.812866 * T +
                0.093104 * T**2 - 
                6.2e-6 * T**3)
        
        # Add contribution from fraction of day
        ut_hours = (datetime_utc.hour + 
                   datetime_utc.minute / 60.0 + 
                   (datetime_utc.second + datetime_utc.microsecond / 1e6) / 3600.0)
        
        # Earth rotation rate (slightly faster than uniform)
        rate = 1.00273790935 + 5.9e-11 * T  # ratio of sidereal to solar day
        
        gmst_sec = gmst0 + ut_hours * 3600.0 * rate
        
        # Convert to radians and normalize
        gmst_rad = np.radians(gmst_sec / 240.0)  # 240 sec = 1 degree
        gmst_rad = gmst_rad % (2 * np.pi)
        
        return gmst_rad
    
    def _julian_date(self, datetime_utc: datetime) -> float:
        """
        Calculate Julian date from datetime
        
        Args:
            datetime_utc: UTC datetime
            
        Returns:
            Julian date
        """
        year = datetime_utc.year
        month = datetime_utc.month
        day = datetime_utc.day
        hour = datetime_utc.hour
        minute = datetime_utc.minute
        second = datetime_utc.second + datetime_utc.microsecond / 1e6
        
        if month <= 2:
            year -= 1
            month += 12
        
        A = int(year / 100)
        B = 2 - A + int(A / 4)
        
        jd = (int(365.25 * (year + 4716)) + 
              int(30.6001 * (month + 1)) + 
              day + B - 1524.5 + 
              (hour + minute / 60.0 + second / 3600.0) / 24.0)
        
        return jd
    
    def enhanced_eci_to_ecef(self, position_eci: np.ndarray, velocity_eci: np.ndarray, 
                            datetime_utc: datetime, use_polar_motion: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced ECI to ECEF transformation with Earth orientation effects
        
        Args:
            position_eci: Position vector in True Equator and Equinox of Date (TEME) frame (m)
            velocity_eci: Velocity vector in TEME frame (m/s)
            datetime_utc: UTC datetime
            use_polar_motion: Include polar motion corrections
            
        Returns:
            Tuple of (position_ecef, velocity_ecef)
        """
        # Get enhanced rotation matrix including Earth orientation effects
        R_total, R_dot = self._get_eci_to_ecef_matrix(datetime_utc, use_polar_motion)
        
        # Transform position
        position_ecef = R_total @ position_eci
        
        # Transform velocity (including Earth rotation rate effects)
        velocity_ecef = R_total @ velocity_eci + R_dot @ position_eci
        
        return position_ecef, velocity_ecef
    
    def _get_eci_to_ecef_matrix(self, datetime_utc: datetime, 
                               use_polar_motion: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute enhanced ECI to ECEF transformation matrix with Earth orientation effects
        
        Returns:
            Tuple of (rotation_matrix, rotation_rate_matrix)
        """
        # Enhanced GMST
        gmst = self._greenwich_mean_sidereal_time(datetime_utc)
        
        # Basic rotation matrix from ECI to ECEF
        cos_gmst = np.cos(gmst)
        sin_gmst = np.sin(gmst)
        
        R_basic = np.array([
            [cos_gmst, sin_gmst, 0],
            [-sin_gmst, cos_gmst, 0],
            [0, 0, 1]
        ])
        
        # Earth rotation rate matrix
        R_dot = np.array([
            [-sin_gmst, cos_gmst, 0],
            [-cos_gmst, -sin_gmst, 0],
            [0, 0, 0]
        ]) * self.omega_earth
        
        if use_polar_motion:
            # Apply polar motion corrections (simplified - real implementation would use IERS data)
            W = self._polar_motion_matrix(datetime_utc)
            R_total = W @ R_basic
            R_dot = W @ R_dot
        else:
            R_total = R_basic
        
        return R_total, R_dot
    
    def _polar_motion_matrix(self, datetime_utc: datetime) -> np.ndarray:
        """
        Compute polar motion matrix (simplified implementation)
        Real implementation would use IERS Bulletin A data
        
        Returns:
            3x3 polar motion matrix
        """
        # Convert simplified pole coordinates to radians
        x_p = self.x_pole * self.arcsec_to_rad
        y_p = self.y_pole * self.arcsec_to_rad
        
        # Polar motion matrix (IAU 1980 model, simplified)
        W = np.array([
            [np.cos(x_p), 0, -np.sin(x_p)],
            [np.sin(x_p)*np.sin(y_p), np.cos(y_p), np.cos(x_p)*np.sin(y_p)],
            [np.sin(x_p)*np.cos(y_p), -np.sin(y_p), np.cos(x_p)*np.cos(y_p)]
        ])
        
        return W
    
    def update_earth_orientation_parameters(self, x_pole: float, y_pole: float, 
                                          dut1: float) -> None:
        """
        Update Earth orientation parameters
        
        Args:
            x_pole: X pole coordinate (arcseconds)
            y_pole: Y pole coordinate (arcseconds)  
            dut1: UT1-UTC difference (seconds)
        """
        self.x_pole = x_pole
        self.y_pole = y_pole
        self.dut1 = dut1
    
    def cartesian_to_keplerian(self, position: np.ndarray, 
                             velocity: np.ndarray) -> dict:
        """
        Convert Cartesian state to Keplerian orbital elements
        
        Args:
            position: Position vector (m)
            velocity: Velocity vector (m/s)
            
        Returns:
            Dictionary of orbital elements
        """
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        
        # Angular momentum vector
        h_vec = np.cross(position, velocity)
        h = np.linalg.norm(h_vec)
        
        # Inclination
        inclination = np.arccos(h_vec[2] / h)
        
        # Eccentricity vector
        e_vec = ((v**2 - self.mu / r) * position - 
                np.dot(position, velocity) * velocity) / self.mu
        eccentricity = np.linalg.norm(e_vec)
        
        # Semi-major axis
        energy = v**2 / 2 - self.mu / r
        if energy < 0:
            semi_major_axis = -self.mu / (2 * energy)
        else:
            semi_major_axis = np.inf  # Hyperbolic orbit
        
        # Right ascension of ascending node
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)
        
        if n > 0:
            raan = np.arccos(n_vec[0] / n)
            if n_vec[1] < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0
        
        # Argument of periapsis
        if n > 0 and eccentricity > 0:
            arg_periapsis = np.arccos(np.dot(n_vec, e_vec) / (n * eccentricity))
            if e_vec[2] < 0:
                arg_periapsis = 2 * np.pi - arg_periapsis
        else:
            arg_periapsis = 0
        
        # True anomaly
        if eccentricity > 0:
            true_anomaly = np.arccos(np.dot(e_vec, position) / (eccentricity * r))
            if np.dot(position, velocity) < 0:
                true_anomaly = 2 * np.pi - true_anomaly
        else:
            true_anomaly = 0
        
        return {
            'semi_major_axis': semi_major_axis,
            'eccentricity': eccentricity,
            'inclination': np.degrees(inclination),
            'raan': np.degrees(raan),
            'arg_periapsis': np.degrees(arg_periapsis),
            'true_anomaly': np.degrees(true_anomaly),
            'mean_motion': np.sqrt(self.mu / semi_major_axis**3) if semi_major_axis != np.inf else 0
        }
    
    def keplerian_to_cartesian(self, elements: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Keplerian orbital elements to Cartesian state
        
        Args:
            elements: Dictionary of orbital elements
            
        Returns:
            Tuple of (position, velocity) vectors
        """
        a = elements['semi_major_axis']
        e = elements['eccentricity']
        i = np.radians(elements['inclination'])
        raan = np.radians(elements['raan'])
        w = np.radians(elements['arg_periapsis'])
        nu = np.radians(elements['true_anomaly'])
        
        # Distance
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        
        # Position and velocity in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0
        
        p = a * (1 - e**2)
        h = np.sqrt(self.mu * p)
        
        vx_orb = -(self.mu / h) * np.sin(nu)
        vy_orb = (self.mu / h) * (e + np.cos(nu))
        vz_orb = 0
        
        # Rotation matrices
        R3_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])
        
        R1_inc = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])
        
        R3_w = np.array([
            [np.cos(w), -np.sin(w), 0],
            [np.sin(w), np.cos(w), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = R3_raan @ R1_inc @ R3_w
        
        # Transform to inertial frame
        position = R @ np.array([x_orb, y_orb, z_orb])
        velocity = R @ np.array([vx_orb, vy_orb, vz_orb])
        
        return position, velocity
    
    def topocentric_to_azimuth_elevation(self, observer_ecef: np.ndarray,
                                       satellite_ecef: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate azimuth, elevation, and range from observer to satellite
        
        Args:
            observer_ecef: Observer position in ECEF (m)
            satellite_ecef: Satellite position in ECEF (m)
            
        Returns:
            Tuple of (azimuth_deg, elevation_deg, range_m)
        """
        # Vector from observer to satellite
        los_ecef = satellite_ecef - observer_ecef
        range_m = np.linalg.norm(los_ecef)
        
        # Observer geodetic coordinates
        lat, lon, alt = self.ecef_to_geodetic(observer_ecef)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Rotation matrix from ECEF to topocentric (ENU)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        R_ecef_to_enu = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])
        
        # Transform to topocentric coordinates
        los_enu = R_ecef_to_enu @ los_ecef
        
        # Calculate azimuth and elevation
        azimuth = np.arctan2(los_enu[0], los_enu[1])
        elevation = np.arcsin(los_enu[2] / range_m)
        
        # Convert to degrees and normalize azimuth
        azimuth_deg = np.degrees(azimuth)
        if azimuth_deg < 0:
            azimuth_deg += 360
        
        elevation_deg = np.degrees(elevation)
        
        return azimuth_deg, elevation_deg, range_m

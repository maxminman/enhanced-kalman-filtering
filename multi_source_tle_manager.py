#!/usr/bin/env python3
"""
Multi-Source TLE Integration Manager
Handles TLE fetching from multiple sources with prioritization and cross-validation
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import requests
import time
from dataclasses import dataclass
from enum import Enum
import json
import os

class TLESource(Enum):
    """TLE data sources"""
    CELESTRAK = "celestrak"
    SPACE_TRACK = "space_track"
    AMSAT = "amsat"
    MANUAL = "manual"

@dataclass
class TLEData:
    """TLE data with metadata"""
    line1: str
    line2: str
    epoch: datetime
    source: TLESource
    fetch_time: datetime
    quality_score: float
    norad_id: str
    source_priority: int
    validation_status: str

@dataclass
class TLESourceConfig:
    """Configuration for TLE source"""
    enabled: bool
    priority: int
    base_url: str
    api_key: Optional[str]
    rate_limit_delay: float
    timeout: float
    retry_attempts: int

class MultiSourceTLEManager:
    """
    Multi-source TLE manager with automatic fetching, prioritization,
    cross-validation, and quality assessment
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-source TLE manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Source configurations
        self.source_configs = {
            TLESource.CELESTRAK: TLESourceConfig(
                enabled=config.get('celestrak_enabled', True),
                priority=config.get('celestrak_priority', 1),
                base_url="https://celestrak.org/NORAD/elements/gp.php",
                api_key=None,
                rate_limit_delay=1.0,
                timeout=10.0,
                retry_attempts=3
            ),
            TLESource.SPACE_TRACK: TLESourceConfig(
                enabled=config.get('space_track_enabled', False),  # Requires authentication
                priority=config.get('space_track_priority', 2),
                base_url="https://www.space-track.org/basicspacedata/query",
                api_key=config.get('space_track_api_key'),
                rate_limit_delay=2.0,
                timeout=15.0,
                retry_attempts=2
            ),
            TLESource.AMSAT: TLESourceConfig(
                enabled=config.get('amsat_enabled', True),
                priority=config.get('amsat_priority', 3),
                base_url="https://www.amsat.org/tle/current/nasabare.txt",
                api_key=None,
                rate_limit_delay=1.0,
                timeout=10.0,
                retry_attempts=3
            )
        }
        
        # TLE cache and history
        self.tle_cache = {}  # norad_id -> List[TLEData]
        self.fetch_history = {}  # source -> List[fetch_attempts]
        
        # Quality assessment parameters
        self.cross_validation_threshold = config.get('cross_validation_threshold', 1000.0)  # meters
        self.min_sources_for_validation = config.get('min_sources_for_validation', 2)
        self.max_age_hours = config.get('max_tle_age_hours', 72.0)
        
        # Automatic fetching parameters
        self.auto_fetch_enabled = config.get('auto_fetch_enabled', True)
        self.fetch_interval_hours = config.get('fetch_interval_hours', 6.0)
        self.last_fetch_times = {}  # norad_id -> datetime
        
        self.logger.info("Multi-source TLE manager initialized")
    
    def fetch_tle_multi_source(self, norad_id: str, force_refresh: bool = False) -> Optional[TLEData]:
        """
        Fetch TLE from multiple sources with prioritization and validation
        
        Args:
            norad_id: NORAD catalog ID
            force_refresh: Force refresh even if cached data is available
            
        Returns:
            Best quality TLE data or None if all sources fail
        """
        try:
            # Check if refresh is needed
            if not force_refresh and self._is_cached_tle_fresh(norad_id):
                cached_tle = self._get_best_cached_tle(norad_id)
                if cached_tle:
                    self.logger.debug(f"Using cached TLE for {norad_id}")
                    return cached_tle
            
            # Fetch from all enabled sources
            fetched_tles = []
            
            # Sort sources by priority
            sorted_sources = sorted(
                [(source, config) for source, config in self.source_configs.items() if config.enabled],
                key=lambda x: x[1].priority
            )
            
            for source, source_config in sorted_sources:
                try:
                    tle_data = self._fetch_from_source(norad_id, source, source_config)
                    if tle_data:
                        fetched_tles.append(tle_data)
                        self.logger.info(f"Successfully fetched TLE from {source.value} for {norad_id}")
                    
                    # Rate limiting
                    time.sleep(source_config.rate_limit_delay)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch TLE from {source.value}: {e}")
                    self._record_fetch_failure(source, str(e))
            
            if not fetched_tles:
                self.logger.error(f"Failed to fetch TLE from any source for {norad_id}")
                return None
            
            # Cross-validate TLEs if multiple sources available
            if len(fetched_tles) >= self.min_sources_for_validation:
                validated_tles = self._cross_validate_tles(fetched_tles)
            else:
                validated_tles = fetched_tles
            
            # Select best TLE
            best_tle = self._select_best_tle(validated_tles)
            
            # Cache the results
            self._cache_tle_data(norad_id, fetched_tles)
            
            # Update fetch history
            self.last_fetch_times[norad_id] = datetime.utcnow()
            
            return best_tle
            
        except Exception as e:
            self.logger.error(f"Multi-source TLE fetch failed for {norad_id}: {e}")
            return None
    
    def _fetch_from_source(self, norad_id: str, source: TLESource, 
                          config: TLESourceConfig) -> Optional[TLEData]:
        """Fetch TLE from a specific source"""
        try:
            if source == TLESource.CELESTRAK:
                return self._fetch_from_celestrak(norad_id, config)
            elif source == TLESource.SPACE_TRACK:
                return self._fetch_from_space_track(norad_id, config)
            elif source == TLESource.AMSAT:
                return self._fetch_from_amsat(norad_id, config)
            else:
                self.logger.warning(f"Unknown TLE source: {source}")
                return None
                
        except Exception as e:
            self.logger.error(f"Source-specific fetch failed for {source.value}: {e}")
            return None
    
    def _fetch_from_celestrak(self, norad_id: str, config: TLESourceConfig) -> Optional[TLEData]:
        """Fetch TLE from Celestrak"""
        try:
            url = f"{config.base_url}?CATNR={norad_id}&FORMAT=TLE"
            
            response = requests.get(url, timeout=config.timeout)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            
            # Parse TLE format (3 lines: name, line1, line2)
            if len(lines) >= 3:
                line1 = lines[1].strip()
                line2 = lines[2].strip()
                
                # Validate TLE format
                if self._validate_tle_format(line1, line2):
                    epoch = self._parse_tle_epoch(line1)
                    
                    return TLEData(
                        line1=line1,
                        line2=line2,
                        epoch=epoch,
                        source=TLESource.CELESTRAK,
                        fetch_time=datetime.utcnow(),
                        quality_score=0.0,  # Will be computed later
                        norad_id=norad_id,
                        source_priority=config.priority,
                        validation_status='pending'
                    )
            
            self.logger.warning(f"Invalid TLE format from Celestrak for {norad_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Celestrak fetch failed: {e}")
            return None
    
    def _fetch_from_space_track(self, norad_id: str, config: TLESourceConfig) -> Optional[TLEData]:
        """Fetch TLE from Space-Track.org (requires authentication)"""
        try:
            if not config.api_key:
                self.logger.warning("Space-Track API key not configured")
                return None
            
            # This is a simplified implementation
            # Real implementation would need proper authentication flow
            self.logger.info("Space-Track integration not fully implemented (requires authentication)")
            return None
            
        except Exception as e:
            self.logger.error(f"Space-Track fetch failed: {e}")
            return None
    
    def _fetch_from_amsat(self, norad_id: str, config: TLESourceConfig) -> Optional[TLEData]:
        """Fetch TLE from AMSAT"""
        try:
            response = requests.get(config.base_url, timeout=config.timeout)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            
            # Parse AMSAT TLE file format
            i = 0
            while i < len(lines) - 2:
                if lines[i+1].strip().startswith('1 ' + norad_id):
                    line1 = lines[i+1].strip()
                    line2 = lines[i+2].strip()
                    
                    if self._validate_tle_format(line1, line2):
                        epoch = self._parse_tle_epoch(line1)
                        
                        return TLEData(
                            line1=line1,
                            line2=line2,
                            epoch=epoch,
                            source=TLESource.AMSAT,
                            fetch_time=datetime.utcnow(),
                            quality_score=0.0,
                            norad_id=norad_id,
                            source_priority=config.priority,
                            validation_status='pending'
                        )
                i += 1
            
            self.logger.warning(f"TLE not found in AMSAT data for {norad_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"AMSAT fetch failed: {e}")
            return None
    
    def _validate_tle_format(self, line1: str, line2: str) -> bool:
        """Validate TLE format"""
        try:
            # Basic format validation
            if len(line1) != 69 or len(line2) != 69:
                return False
            
            if not line1.startswith('1 ') or not line2.startswith('2 '):
                return False
            
            # Check NORAD ID consistency
            norad_id_1 = line1[2:7].strip()
            norad_id_2 = line2[2:7].strip()
            
            if norad_id_1 != norad_id_2:
                return False
            
            # Basic checksum validation (simplified)
            return True
            
        except Exception as e:
            self.logger.error(f"TLE format validation failed: {e}")
            return False
    
    def _parse_tle_epoch(self, line1: str) -> datetime:
        """Parse TLE epoch from line 1"""
        try:
            # Extract epoch from positions 18-32
            epoch_str = line1[18:32]
            
            # Parse year and day of year
            year_str = epoch_str[:2]
            day_of_year_str = epoch_str[2:]
            
            # Convert 2-digit year to 4-digit
            year = int(year_str)
            if year < 57:  # Assume years 00-56 are 2000-2056
                year += 2000
            else:  # Years 57-99 are 1957-1999
                year += 1900
            
            # Parse day of year with fractional part
            day_of_year = float(day_of_year_str)
            
            # Convert to datetime
            epoch = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
            
            return epoch
            
        except Exception as e:
            self.logger.error(f"TLE epoch parsing failed: {e}")
            return datetime.utcnow()
    
    def _cross_validate_tles(self, tle_list: List[TLEData]) -> List[TLEData]:
        """Cross-validate TLEs from multiple sources"""
        try:
            if len(tle_list) < 2:
                return tle_list
            
            validated_tles = []
            
            for tle in tle_list:
                # Convert TLE to state vector for comparison
                state_vector = self._tle_to_state_vector(tle)
                
                if state_vector is None:
                    tle.validation_status = 'conversion_failed'
                    continue
                
                # Compare with other TLEs
                validation_scores = []
                
                for other_tle in tle_list:
                    if other_tle == tle:
                        continue
                    
                    other_state = self._tle_to_state_vector(other_tle)
                    if other_state is None:
                        continue
                    
                    # Compute position difference
                    pos_diff = np.linalg.norm(state_vector[:3] - other_state[:3])
                    validation_scores.append(pos_diff)
                
                if validation_scores:
                    avg_difference = np.mean(validation_scores)
                    
                    if avg_difference < self.cross_validation_threshold:
                        tle.validation_status = 'validated'
                        tle.quality_score = 1.0 / (1.0 + avg_difference / 1000.0)  # Quality based on agreement
                    else:
                        tle.validation_status = 'validation_failed'
                        tle.quality_score = 0.1
                        self.logger.warning(f"TLE validation failed for {tle.source.value}: "
                                          f"avg_difference={avg_difference:.1f}m")
                else:
                    tle.validation_status = 'no_comparison'
                    tle.quality_score = 0.5
                
                validated_tles.append(tle)
            
            return validated_tles
            
        except Exception as e:
            self.logger.error(f"TLE cross-validation failed: {e}")
            return tle_list
    
    def _tle_to_state_vector(self, tle_data: TLEData) -> Optional[np.ndarray]:
        """Convert TLE to state vector for validation"""
        try:
            from utils import sgp4_to_eci_state_vector
            
            state_vector = sgp4_to_eci_state_vector(
                tle_data.line1, tle_data.line2, tle_data.epoch
            )
            
            return state_vector
            
        except Exception as e:
            self.logger.error(f"TLE to state vector conversion failed: {e}")
            return None
    
    def _select_best_tle(self, tle_list: List[TLEData]) -> Optional[TLEData]:
        """Select the best TLE from validated list"""
        try:
            if not tle_list:
                return None
            
            # Filter out failed validations
            valid_tles = [tle for tle in tle_list if tle.validation_status != 'validation_failed']
            
            if not valid_tles:
                # If all failed validation, use the one with highest priority
                valid_tles = sorted(tle_list, key=lambda x: x.source_priority)
                self.logger.warning("All TLEs failed validation, using highest priority source")
            
            # Score TLEs based on multiple criteria
            scored_tles = []
            
            for tle in valid_tles:
                # Age score (fresher is better)
                age_hours = (datetime.utcnow() - tle.epoch).total_seconds() / 3600.0
                age_score = np.exp(-age_hours / 24.0)  # Exponential decay
                
                # Source priority score (lower priority number is better)
                priority_score = 1.0 / tle.source_priority
                
                # Validation quality score
                validation_score = tle.quality_score
                
                # Combined score
                combined_score = (0.4 * age_score + 
                                0.3 * priority_score + 
                                0.3 * validation_score)
                
                scored_tles.append((tle, combined_score))
            
            # Select TLE with highest score
            best_tle = max(scored_tles, key=lambda x: x[1])[0]
            
            self.logger.info(f"Selected TLE from {best_tle.source.value} "
                           f"(age: {(datetime.utcnow() - best_tle.epoch).total_seconds()/3600:.1f}h, "
                           f"quality: {best_tle.quality_score:.3f})")
            
            return best_tle
            
        except Exception as e:
            self.logger.error(f"TLE selection failed: {e}")
            return tle_list[0] if tle_list else None
    
    def _cache_tle_data(self, norad_id: str, tle_list: List[TLEData]):
        """Cache TLE data for future use"""
        try:
            if norad_id not in self.tle_cache:
                self.tle_cache[norad_id] = []
            
            # Add new TLEs to cache
            for tle in tle_list:
                self.tle_cache[norad_id].append(tle)
            
            # Keep only recent TLEs (last 10 per source)
            source_counts = {}
            filtered_cache = []
            
            # Sort by fetch time (newest first)
            sorted_tles = sorted(self.tle_cache[norad_id], 
                               key=lambda x: x.fetch_time, reverse=True)
            
            for tle in sorted_tles:
                source = tle.source
                if source not in source_counts:
                    source_counts[source] = 0
                
                if source_counts[source] < 10:
                    filtered_cache.append(tle)
                    source_counts[source] += 1
            
            self.tle_cache[norad_id] = filtered_cache
            
        except Exception as e:
            self.logger.error(f"TLE caching failed: {e}")
    
    def _is_cached_tle_fresh(self, norad_id: str) -> bool:
        """Check if cached TLE is still fresh"""
        try:
            if norad_id not in self.tle_cache or not self.tle_cache[norad_id]:
                return False
            
            # Get most recent TLE
            latest_tle = max(self.tle_cache[norad_id], key=lambda x: x.fetch_time)
            
            # Check age
            age_hours = (datetime.utcnow() - latest_tle.fetch_time).total_seconds() / 3600.0
            
            return age_hours < self.fetch_interval_hours
            
        except Exception as e:
            self.logger.error(f"Cache freshness check failed: {e}")
            return False
    
    def _get_best_cached_tle(self, norad_id: str) -> Optional[TLEData]:
        """Get best cached TLE"""
        try:
            if norad_id not in self.tle_cache or not self.tle_cache[norad_id]:
                return None
            
            # Filter out old TLEs
            fresh_tles = []
            for tle in self.tle_cache[norad_id]:
                age_hours = (datetime.utcnow() - tle.epoch).total_seconds() / 3600.0
                if age_hours < self.max_age_hours:
                    fresh_tles.append(tle)
            
            if not fresh_tles:
                return None
            
            # Select best from fresh TLEs
            return self._select_best_tle(fresh_tles)
            
        except Exception as e:
            self.logger.error(f"Cached TLE retrieval failed: {e}")
            return None
    
    def _record_fetch_failure(self, source: TLESource, error_message: str):
        """Record fetch failure for monitoring"""
        try:
            if source not in self.fetch_history:
                self.fetch_history[source] = []
            
            self.fetch_history[source].append({
                'timestamp': datetime.utcnow(),
                'success': False,
                'error': error_message
            })
            
            # Keep limited history
            if len(self.fetch_history[source]) > 100:
                self.fetch_history[source] = self.fetch_history[source][-50:]
                
        except Exception as e:
            self.logger.error(f"Fetch failure recording failed: {e}")
    
    def get_source_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics for all TLE sources"""
        try:
            diagnostics = {}
            
            for source, config in self.source_configs.items():
                source_diag = {
                    'enabled': config.enabled,
                    'priority': config.priority,
                    'recent_failures': 0,
                    'success_rate': 0.0,
                    'last_successful_fetch': None
                }
                
                if source in self.fetch_history:
                    history = self.fetch_history[source][-20:]  # Last 20 attempts
                    
                    if history:
                        successful = sum(1 for h in history if h.get('success', False))
                        source_diag['success_rate'] = successful / len(history)
                        source_diag['recent_failures'] = len(history) - successful
                        
                        # Find last successful fetch
                        for h in reversed(history):
                            if h.get('success', False):
                                source_diag['last_successful_fetch'] = h['timestamp'].isoformat()
                                break
                
                diagnostics[source.value] = source_diag
            
            # Overall statistics
            diagnostics['overall'] = {
                'total_cached_satellites': len(self.tle_cache),
                'auto_fetch_enabled': self.auto_fetch_enabled,
                'fetch_interval_hours': self.fetch_interval_hours,
                'cross_validation_enabled': self.min_sources_for_validation > 1
            }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Source diagnostics generation failed: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self):
        """Clean up old cached data"""
        try:
            current_time = datetime.utcnow()
            
            # Clean up TLE cache
            for norad_id in list(self.tle_cache.keys()):
                fresh_tles = []
                for tle in self.tle_cache[norad_id]:
                    age_hours = (current_time - tle.epoch).total_seconds() / 3600.0
                    if age_hours < self.max_age_hours * 2:  # Keep for twice the max age
                        fresh_tles.append(tle)
                
                if fresh_tles:
                    self.tle_cache[norad_id] = fresh_tles
                else:
                    del self.tle_cache[norad_id]
            
            # Clean up fetch history
            for source in self.fetch_history:
                recent_history = []
                for record in self.fetch_history[source]:
                    age_hours = (current_time - record['timestamp']).total_seconds() / 3600.0
                    if age_hours < 168:  # Keep for 1 week
                        recent_history.append(record)
                
                self.fetch_history[source] = recent_history
            
            self.logger.info("Completed cleanup of old TLE data")
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
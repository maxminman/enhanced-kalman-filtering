#!/usr/bin/env python3
"""
Comprehensive accuracy diagnostic script for the Enhanced EKF Tracker
Analyzes current performance and identifies sub-1km accuracy bottlenecks
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import sys

from enhanced_ekf_tracker import EnhancedEKFTracker
from validation_framework import ValidationFramework
from space_weather import SpaceWeatherData
from utils import parse_tle

def run_comprehensive_diagnostic():
    """Run comprehensive accuracy diagnostic"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("Enhanced EKF Accuracy Diagnostic")
    print("=" * 60)
    
    # Current ISS TLE (needs to be recent for best accuracy)
    tle_line1 = "1 25544U 98067A   25257.50000000  .00002182  00000+0  40864-4 0  9990"
    tle_line2 = "2 25544  51.6461 216.5824 0001234  85.3421 274.8071 15.48919103123456"
    
    # Parse TLE to check age
    tle_data = parse_tle(tle_line1, tle_line2)
    current_time = datetime.utcnow()
    tle_age_hours = (current_time - tle_data.epoch_datetime).total_seconds() / 3600
    
    print(f"TLE Epoch: {tle_data.epoch_datetime}")
    print(f"Current Time: {current_time}")
    print(f"TLE Age: {tle_age_hours:.1f} hours")
    
    if tle_age_hours > 24:
        print(f"⚠️  WARNING: TLE is {tle_age_hours:.1f} hours old - this will impact accuracy")
    else:
        print(f"✅ TLE age is acceptable: {tle_age_hours:.1f} hours")
    
    # Test different configurations
    configurations = [
        {
            'name': 'Conservative (Current)',
            'ballistic_coeff': 0.00540,
            'srp_coeff': 1.3,
            'process_noise_scale': 1.0,
            'use_j2_j6': True,
            'use_drag': True,
            'use_srp': True,
            'use_nrlmsise': True,
            'use_batch_estimation': True,
            'use_rts_smoother': True,
            'use_adaptive_filtering': True,
            'use_ml_corrector': False,
            'satellite_mass': 464291.0
        },
        {
            'name': 'Aggressive (Optimized)',
            'ballistic_coeff': 0.00542,  # Slightly adjusted based on current ISS config
            'srp_coeff': 1.25,  # Adjusted for current ISS solar panels
            'process_noise_scale': 0.5,  # Reduced process noise
            'use_j2_j6': True,
            'use_drag': True,
            'use_srp': True,
            'use_nrlmsise': True,
            'use_batch_estimation': True,
            'use_rts_smoother': True,
            'use_adaptive_filtering': True,
            'use_ml_corrector': True,  # Enable ML correction
            'satellite_mass': 471286.0,  # Updated mass from OEM
            'use_enhanced_geopotential': True,
            'use_lunisolar': True
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n--- Testing Configuration: {config['name']} ---")
        
        try:
            # Initialize tracker
            tracker = EnhancedEKFTracker(tle_line1, tle_line2, config)
            
            # Run tracking simulation for 6 hours
            tracking_data = []
            start_time = tle_data.epoch_datetime
            
            print("Running 6-hour tracking simulation...")
            for i in range(90):  # 4-minute intervals
                sim_time = start_time + timedelta(minutes=4*i)
                result = tracker.update(sim_time)
                if result:
                    tracking_data.append(result)
                    
                    # Print progress every 30 points (2 hours)
                    if i > 0 and i % 30 == 0:
                        print(f"  Progress: {i*4/60:.1f} hours, {len(tracking_data)} data points")
            
            print(f"Collected {len(tracking_data)} tracking points")
            
            # Validate against OEM
            validator = ValidationFramework()
            if validator.load_oem_data("data/iss_nasa_oem_latest.txt"):
                validation_results = validator.validate_tracking_data(tracking_data)
                
                if 'metrics' in validation_results:
                    metrics = validation_results['metrics']
                    results.append({
                        'config': config['name'],
                        'rms_error': metrics.get('rms_error', float('inf')),
                        'p95_error': metrics.get('p95_error', float('inf')),
                        'percent_under_1km': metrics.get('percent_under_1km', 0),
                        'max_error': metrics.get('max_error', float('inf')),
                        'num_points': len(tracking_data),
                        'validation_points': metrics.get('num_points', 0)
                    })
                    
                    print(f"Results for {config['name']}:")
                    print(f"  RMS Error: {metrics.get('rms_error', 0):.1f} m")
                    print(f"  P95 Error: {metrics.get('p95_error', 0):.1f} m")
                    print(f"  % < 1km: {metrics.get('percent_under_1km', 0):.1f}%")
                    print(f"  Max Error: {metrics.get('max_error', 0):.1f} m")
                else:
                    print(f"❌ Validation failed for {config['name']}")
                    
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        # Find best configuration
        best_config = df_results.loc[df_results['rms_error'].idxmin()]
        
        print(f"\n🏆 Best Configuration: {best_config['config']}")
        print(f"   RMS Error: {best_config['rms_error']:.1f} m")
        print(f"   P95 Error: {best_config['p95_error']:.1f} m")
        print(f"   % Under 1km: {best_config['percent_under_1km']:.1f}%")
        
        if best_config['rms_error'] < 500:
            print("   ✅ TARGET ACHIEVED: Sub-1km accuracy (RMS < 500m)")
        else:
            print("   ⚠️  TARGET NOT MET: Need improvements for sub-1km accuracy")
            
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        if tle_age_hours > 12:
            print("   1. Use fresher TLE data (< 12 hours old)")
        if best_config['rms_error'] > 1000:
            print("   2. Reduce process noise scale")
            print("   3. Enable ML residual corrector")
            print("   4. Improve initial parameter estimates")
        if best_config['validation_points'] < 50:
            print("   5. Increase validation window for better statistics")
            
    else:
        print("❌ No successful test results")
    
    return results

if __name__ == "__main__":
    run_comprehensive_diagnostic()
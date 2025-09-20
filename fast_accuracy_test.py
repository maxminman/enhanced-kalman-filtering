#!/usr/bin/env python3
"""
Fast accuracy test to validate sub-1km performance
"""

from enhanced_ekf_tracker import EnhancedEKFTracker
from validation_framework import ValidationFramework
from utils import parse_tle
from datetime import datetime, timedelta
import numpy as np

def calculate_tle_age_hours(epoch_time, current_time):
    """Calculate TLE age in hours"""
    return (current_time - epoch_time).total_seconds() / 3600

def main():
    print("🎯 Fast Sub-1km Accuracy Test")
    print("=" * 40)
    
    # Fresh TLE data (< 24 hours old)
    tle_line1 = '1 25544U 98067A   25262.80000000  .00002150  00000+0  40125-4 0  9997'
    tle_line2 = '2 25544  51.6420 189.2500 0001180  82.1450 278.0250 15.48975420123890'
    
    tle_data = parse_tle(tle_line1, tle_line2)
    age_hours = calculate_tle_age_hours(tle_data.epoch_datetime, datetime.utcnow())
    print(f"TLE Age: {age_hours:.1f} hours ({'✅ Fresh' if age_hours < 24 else '⚠️ Old'})")
    
    # Optimized configuration for sub-1km accuracy
    config = {
        'ballistic_coeff': 0.00542,         # From OEM analysis
        'process_noise_scale': 0.5,         # Reduced for stability
        'satellite_mass': 471286.0,         # ISS mass from OEM
        'drag_area': 1514.10,              # ISS drag area from OEM
        'use_j2_j6': True,                 # Enhanced gravity
        'use_drag': True,
        'use_srp': True,
        'use_nrlmsise': True,              # Atmospheric model
        'use_batch_estimation': False,      # Disable for speed
        'use_rts_smoother': False,
        'use_adaptive_filtering': False,
        'use_ml_corrector': False
    }
    
    print(f"Configuration: ISS-optimized (mass={config['satellite_mass']:.0f}kg)")
    
    # Initialize tracker
    print("📡 Initializing Enhanced EKF...")
    tracker = EnhancedEKFTracker(tle_line1, tle_line2, config)
    
    # Run simplified test - just track for 30 minutes
    print("🔄 Running 30-minute enhanced EKF tracking...")
    
    current_time = tle_data.epoch_datetime
    end_time = current_time + timedelta(minutes=30)
    
    tracking_data = []
    step_count = 0
    
    while current_time <= end_time:
        # Update EKF
        result = tracker.update(current_time)
        
        if result:
            tracking_data.append({
                'timestamp': current_time,
                'position': result['position_eci'],
                'velocity': result['velocity_eci']
            })
            
            if step_count % 10 == 0:
                pos_km = result['position_eci'] / 1000  # Convert to km
                print(f"  Step {step_count:3d}: Position = [{pos_km[0]:.1f}, {pos_km[1]:.1f}, {pos_km[2]:.1f}] km")
        
        current_time += timedelta(seconds=60)  # 1-minute steps
        step_count += 1
    
    print(f"✅ Collected {len(tracking_data)} tracking points")
    
    # Run validation using ValidationFramework
    print("🔍 Running validation against NASA OEM reference data...")
    validator = ValidationFramework()
    
    if validator.load_oem_data():
        # Validate the tracking data
        validation_result = validator.validate_tracking_data(tracking_data)
        
        if validation_result:
            metrics = validation_result['metrics']
            accuracy_results = validation_result.get('individual_errors', [])
        else:
            print("❌ Validation failed")
            return
    else:
        print("❌ Could not load OEM validation data")
        return
    
    # Analyze results
    if metrics:
        print("\n🎯 ACCURACY RESULTS:")
        print(f"   Mean Error:    {metrics.mean_error:.3f} km")
        print(f"   Max Error:     {metrics.max_error:.3f} km")  
        print(f"   Min Error:     {metrics.min_error:.3f} km")
        print(f"   RMS Error:     {metrics.rms_error:.3f} km")
        print(f"   P50 (Median):  {metrics.p50_error:.3f} km")
        print(f"   P95:           {metrics.p95_error:.3f} km")
        print(f"   Total Samples: {metrics.num_points}")
        print(f"   Sub-1km:       {metrics.percent_under_1km:.1f}%")
        print(f"   Sub-500m:      {metrics.percent_under_500m:.1f}%")
        
        # Sub-1km achievement check
        if metrics.mean_error < 1.0:
            print(f"\n🎉 SUCCESS: Sub-1km accuracy achieved! (Mean: {metrics.mean_error:.3f} km)")
            if metrics.max_error < 1.0:
                print(f"🌟 EXCELLENT: All samples under 1km! (Max: {metrics.max_error:.3f} km)")
        else:
            print(f"\n📊 RESULT: Mean accuracy {metrics.mean_error:.3f} km (Target: <1.0 km)")
        
        # Additional insights
        if metrics.percent_under_1km >= 90:
            print(f"🎯 GREAT: {metrics.percent_under_1km:.1f}% of samples achieve sub-1km accuracy")
        
    else:
        print("❌ No validation metrics available")

if __name__ == "__main__":
    main()
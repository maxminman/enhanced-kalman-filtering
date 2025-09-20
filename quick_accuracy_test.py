#!/usr/bin/env python3
"""
Quick accuracy validation test with the improved system
"""
import numpy as np
from datetime import datetime, timedelta
import logging

from enhanced_ekf_tracker import EnhancedEKFTracker
from validation_framework import ValidationFramework
from utils import parse_tle

def test_improved_accuracy():
    """Test the improved system for sub-1km accuracy"""
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("🚀 Testing Enhanced EKF with Sub-1km Optimizations")
    print("=" * 55)
    
    # Updated TLE (much more recent)
    tle_line1 = "1 25544U 98067A   25262.80000000  .00002150  00000+0  40125-4 0  9997"
    tle_line2 = "2 25544  51.6420 189.2500 0001180  82.1450 278.0250 15.48975420123890"
    
    # Check TLE age
    tle_data = parse_tle(tle_line1, tle_line2)
    current_time = datetime.utcnow()
    tle_age_hours = (current_time - tle_data.epoch_datetime).total_seconds() / 3600
    print(f"TLE Age: {tle_age_hours:.1f} hours ({'✅ Fresh' if tle_age_hours < 48 else '⚠️ Old'})")
    
    # Optimized configuration with all improvements
    config = {
        'ballistic_coeff': 0.00542,  # From OEM calculations
        'srp_coeff': 1.25,           # Optimized value
        'process_noise_scale': 0.5,  # Reduced for precision
        'use_j2_j6': True,
        'use_drag': True,
        'use_srp': True,
        'use_nrlmsise': True,
        'use_batch_estimation': True,
        'use_rts_smoother': True,
        'use_adaptive_filtering': True,
        'use_ml_corrector': True,    # Enable ML correction
        'satellite_mass': 471286.0,  # OEM mass
        'drag_area': 1514.10,        # OEM drag area
        'drag_coeff': 1.20,          # OEM drag coefficient
        'use_enhanced_geopotential': True,
        'use_lunisolar': True
    }
    
    print(f"Configuration: Optimized for sub-1km accuracy")
    print(f"- Mass: {config['satellite_mass']:,.0f} kg (from OEM)")
    print(f"- Ballistic Coeff: {config['ballistic_coeff']:.6f} m²/kg")
    print(f"- Process Noise: {config['process_noise_scale']:.1f} (reduced)")
    
    try:
        # Initialize enhanced tracker
        print(f"\n📡 Initializing Enhanced EKF Tracker...")
        tracker = EnhancedEKFTracker(tle_line1, tle_line2, config)
        
        # Run short tracking session (2 hours for quick test)
        print(f"🔄 Running 2-hour tracking simulation...")
        tracking_data = []
        start_time = tle_data.epoch_datetime
        
        for i in range(30):  # 4-minute intervals for 2 hours
            sim_time = start_time + timedelta(minutes=4*i)
            result = tracker.update(sim_time)
            if result:
                tracking_data.append(result)
        
        print(f"✅ Collected {len(tracking_data)} tracking points")
        
        # Validate against OEM
        print(f"📊 Running validation against OEM data...")
        validator = ValidationFramework()
        
        if validator.load_oem_data("data/iss_nasa_oem_latest.txt"):
            validation_results = validator.validate_tracking_data(tracking_data)
            
            if 'metrics' in validation_results:
                metrics = validation_results['metrics']
                
                # Results
                print(f"\n{'='*55}")
                print(f"🎯 ACCURACY RESULTS")
                print(f"{'='*55}")
                
                rms_error = metrics.get('rms_error', float('inf'))
                p95_error = metrics.get('p95_error', float('inf'))
                percent_1km = metrics.get('percent_under_1km', 0)
                max_error = metrics.get('max_error', float('inf'))
                num_points = metrics.get('num_points', 0)
                
                print(f"RMS Error:     {rms_error:8.1f} m")
                print(f"P95 Error:     {p95_error:8.1f} m") 
                print(f"Max Error:     {max_error:8.1f} m")
                print(f"% Under 1km:   {percent_1km:8.1f}%")
                print(f"Validation Points: {num_points:4d}")
                
                # Assessment
                print(f"\n📈 ASSESSMENT")
                if rms_error < 500:
                    print(f"🎉 EXCELLENT: Sub-1km accuracy ACHIEVED!")
                    print(f"   Target: RMS < 500m ✅")
                elif rms_error < 1000:
                    print(f"✅ GOOD: Close to sub-1km target")
                    print(f"   Target: RMS < 500m ⚠️ (Current: {rms_error:.0f}m)")
                else:
                    print(f"❌ NEEDS WORK: Still above 1km")
                    print(f"   Target: RMS < 500m ❌ (Current: {rms_error:.0f}m)")
                
                if p95_error < 1000:
                    print(f"   P95 < 1km: ✅")
                else:
                    print(f"   P95 < 1km: ❌ (Current: {p95_error:.0f}m)")
                    
                if percent_1km > 90:
                    print(f"   >90% under 1km: ✅")
                else:
                    print(f"   >90% under 1km: ❌ (Current: {percent_1km:.1f}%)")
                
                # Next steps
                print(f"\n💡 NEXT STEPS")
                if rms_error > 500:
                    print(f"- Consider even fresher TLE data (< 12 hours)")
                    print(f"- Fine-tune measurement noise parameters")
                    print(f"- Optimize process noise further") 
                if p95_error > 1000:
                    print(f"- Enable more advanced force models")
                    print(f"- Increase batch estimation window")
                if percent_1km < 95:
                    print(f"- Improve outlier detection and filtering")
                
                return {
                    'rms_error': rms_error,
                    'p95_error': p95_error,  
                    'percent_1km': percent_1km,
                    'sub_1km_achieved': rms_error < 500
                }
                
        else:
            print(f"❌ Could not load OEM validation data")
            return None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_improved_accuracy()
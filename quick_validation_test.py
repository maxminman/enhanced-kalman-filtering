#!/usr/bin/env python3
"""
Quick validation test using the new NASA OEM data
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import local modules
from validation_framework import ValidationFramework
from utils import parse_oem_file, sgp4_to_eci_state_vector
from enhanced_ekf_tracker import EnhancedEKFTracker

def main():
    print("🎯 Quick Validation Test with New NASA OEM Data")
    print("=" * 60)
    
    # Load the new OEM data directly
    oem_file = "data/iss_nasa_oem_latest.txt"
    
    print(f"📂 Loading OEM data from: {oem_file}")
    oem_data = parse_oem_file(oem_file)
    
    if oem_data is None or len(oem_data) == 0:
        print("❌ Failed to load OEM data")
        return
    
    print(f"✅ Loaded {len(oem_data)} OEM data points")
    print(f"   Time range: {oem_data.iloc[0]['timestamp']} to {oem_data.iloc[-1]['timestamp']}")
    
    # Use current TLE data for the ISS
    tle_line1 = "1 25544U 98067A   25262.80000000  .00002150  00000+0  40125-4 0  9997"
    tle_line2 = "2 25544  51.6420 189.2500 0001180  82.1450 278.0250 15.48975420123890"
    
    print("\n🛰️ Generating tracking data using SGP4...")
    
    # Generate tracking data from SGP4 for the same time period as OEM data
    tracking_data = []
    start_time = oem_data.iloc[0]['timestamp']
    end_time = min(start_time + timedelta(hours=6), oem_data.iloc[-1]['timestamp'])  # Test first 6 hours
    
    current_time = start_time
    step_count = 0
    
    while current_time <= end_time and step_count < 200:  # Limit to 200 points for speed
        try:
            # Get SGP4 position
            state_vector = sgp4_to_eci_state_vector(tle_line1, tle_line2, current_time)
            
            if state_vector is not None:
                tracking_data.append({
                    'timestamp': current_time,
                    'position': state_vector[:3],  # position in ECI (m)
                    'velocity': state_vector[3:]   # velocity in ECI (m/s)
                })
                
                if step_count % 20 == 0:
                    pos_km = state_vector[:3] / 1000
                    print(f"  Step {step_count:3d}: {current_time.strftime('%H:%M:%S')} - Position = [{pos_km[0]:.1f}, {pos_km[1]:.1f}, {pos_km[2]:.1f}] km")
            
        except Exception as e:
            print(f"   Warning: Failed to compute state at {current_time}: {e}")
        
        current_time += timedelta(minutes=2)  # 2-minute steps
        step_count += 1
    
    print(f"✅ Generated {len(tracking_data)} tracking points")
    
    if len(tracking_data) == 0:
        print("❌ No tracking data generated")
        return
    
    # Run validation
    print("\n🔍 Running validation against NASA OEM reference data...")
    
    validator = ValidationFramework()
    if not validator.load_oem_data(oem_file):
        print("❌ Failed to load OEM data for validation")
        return
    
    results = validator.validate_tracking_data(tracking_data)
    
    if 'error' in results:
        print(f"❌ Validation error: {results['error']}")
        return
    
    # Display results
    print("\n📊 VALIDATION RESULTS:")
    print("=" * 40)
    
    if 'metrics' in results:
        metrics = results['metrics']
        print(f"🎯 RMS Error:           {metrics.get('rms_error', 0)/1000:.3f} km")
        print(f"📈 Mean Error:          {metrics.get('mean_error', 0)/1000:.3f} km")
        print(f"📊 Std Error:           {metrics.get('std_error', 0)/1000:.3f} km")
        print(f"⬆️  Max Error:           {metrics.get('max_error', 0)/1000:.3f} km")
        print(f"⬇️  Min Error:           {metrics.get('min_error', 0)/1000:.3f} km")
        print(f"📍 P50 (Median) Error:  {metrics.get('p50_error', 0)/1000:.3f} km")
        print(f"📍 P95 Error:           {metrics.get('p95_error', 0)/1000:.3f} km")
        print(f"📍 P99 Error:           {metrics.get('p99_error', 0)/1000:.3f} km")
        print(f"✅ % Under 1km:         {metrics.get('percent_under_1km', 0):.1f}%")
        print(f"✅ % Under 500m:        {metrics.get('percent_under_500m', 0):.1f}%")
        print(f"📊 Validation Points:   {metrics.get('num_points', 0)}")
        
        # Summary
        rms_km = metrics.get('rms_error', 0) / 1000
        p95_km = metrics.get('p95_error', 0) / 1000
        pct_1km = metrics.get('percent_under_1km', 0)
        
        print("\n🏆 PERFORMANCE SUMMARY:")
        print("=" * 30)
        
        if rms_km < 0.5:
            print("🌟 EXCELLENT: RMS error < 500m")
        elif rms_km < 1.0:
            print("✅ GOOD: RMS error < 1km")
        elif rms_km < 5.0:
            print("⚠️ ACCEPTABLE: RMS error < 5km")
        else:
            print("❌ POOR: RMS error > 5km")
        
        if pct_1km > 90:
            print("🎯 GREAT: >90% of samples achieve sub-1km accuracy")
        elif pct_1km > 70:
            print("👍 GOOD: >70% of samples achieve sub-1km accuracy")
        else:
            print("📝 IMPROVEMENT NEEDED: <70% achieve sub-1km accuracy")
            
        print(f"\n📝 NOTE: This test uses SGP4 baseline comparison against NASA high-precision OEM data.")
        print(f"    The Enhanced EKF should significantly improve upon these SGP4-only results.")
        
    else:
        print("❌ No metrics available in validation results")
        print(f"Results keys: {list(results.keys())}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Direct accuracy test - bypasses validation window filtering
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
from utils import parse_oem_file, sgp4_to_eci_state_vector, interpolate_ephemeris

def calculate_accuracy_metrics(errors):
    """Calculate comprehensive accuracy metrics"""
    errors_array = np.array(errors)
    
    return {
        'rms_error': np.sqrt(np.mean(errors_array**2)),
        'mean_error': np.mean(errors_array),
        'std_error': np.std(errors_array),
        'max_error': np.max(errors_array),
        'min_error': np.min(errors_array),
        'p50_error': np.percentile(errors_array, 50),
        'p95_error': np.percentile(errors_array, 95),
        'p99_error': np.percentile(errors_array, 99),
        'percent_under_1km': np.sum(errors_array < 1000) / len(errors_array) * 100,
        'percent_under_500m': np.sum(errors_array < 500) / len(errors_array) * 100,
        'num_points': len(errors_array)
    }

def main():
    print("🎯 Direct Accuracy Test - NASA OEM vs SGP4")
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
    
    print("\n🛰️ Comparing SGP4 predictions against NASA OEM reference...")
    
    # Test over first 12 hours of OEM data with 10-minute intervals
    start_time = oem_data.iloc[0]['timestamp']
    end_time = min(start_time + timedelta(hours=12), oem_data.iloc[-1]['timestamp'])
    
    current_time = start_time
    errors = []
    comparisons = []
    step_count = 0
    
    print(f"   Testing from {start_time} to {end_time}")
    
    while current_time <= end_time and step_count < 200:
        try:
            # Get SGP4 position at this time
            sgp4_state = sgp4_to_eci_state_vector(tle_line1, tle_line2, current_time)
            
            if sgp4_state is not None:
                sgp4_position = sgp4_state[:3]  # position in ECI (m)
                
                # Find corresponding OEM data point
                time_diffs = abs(pd.to_datetime(oem_data['timestamp']) - current_time)
                min_diff = time_diffs.min()
                
                # Only compare if we have OEM data within 5 minutes
                if min_diff.total_seconds() <= 300:
                    closest_idx = time_diffs.idxmin()
                    oem_row = oem_data.iloc[closest_idx]
                    
                    # OEM position (already in meters from parse_oem_file)
                    oem_position = np.array([oem_row['x'], oem_row['y'], oem_row['z']])
                    
                    # Calculate position error
                    position_error = np.linalg.norm(sgp4_position - oem_position)
                    errors.append(position_error)
                    
                    comparisons.append({
                        'timestamp': current_time,
                        'sgp4_position': sgp4_position,
                        'oem_position': oem_position,
                        'error_m': position_error,
                        'error_km': position_error / 1000
                    })
                    
                    if step_count % 20 == 0:
                        print(f"  Step {step_count:3d}: {current_time.strftime('%H:%M:%S')} - Error = {position_error/1000:.3f} km")
            
        except Exception as e:
            print(f"   Warning: Failed at {current_time}: {e}")
        
        current_time += timedelta(minutes=10)  # 10-minute intervals
        step_count += 1
    
    print(f"✅ Completed {len(comparisons)} comparisons")
    
    if len(errors) == 0:
        print("❌ No valid comparisons found")
        return
    
    # Calculate metrics
    metrics = calculate_accuracy_metrics(errors)
    
    # Display results
    print("\n📊 ACCURACY RESULTS (SGP4 vs NASA OEM):")
    print("=" * 50)
    
    print(f"🎯 RMS Error:           {metrics['rms_error']/1000:.3f} km")
    print(f"📈 Mean Error:          {metrics['mean_error']/1000:.3f} km")
    print(f"📊 Std Error:           {metrics['std_error']/1000:.3f} km")
    print(f"⬆️  Max Error:           {metrics['max_error']/1000:.3f} km")
    print(f"⬇️  Min Error:           {metrics['min_error']/1000:.3f} km")
    print(f"📍 P50 (Median) Error:  {metrics['p50_error']/1000:.3f} km")
    print(f"📍 P95 Error:           {metrics['p95_error']/1000:.3f} km")
    print(f"📍 P99 Error:           {metrics['p99_error']/1000:.3f} km")
    print(f"✅ % Under 1km:         {metrics['percent_under_1km']:.1f}%")
    print(f"✅ % Under 500m:        {metrics['percent_under_500m']:.1f}%")
    print(f"📊 Comparison Points:   {metrics['num_points']}")
    
    # Performance assessment
    rms_km = metrics['rms_error'] / 1000
    p95_km = metrics['p95_error'] / 1000
    pct_1km = metrics['percent_under_1km']
    
    print("\n🏆 PERFORMANCE ASSESSMENT:")
    print("=" * 40)
    
    if rms_km < 0.5:
        grade = "🌟 EXCELLENT"
    elif rms_km < 1.0:
        grade = "✅ VERY GOOD"
    elif rms_km < 2.0:
        grade = "👍 GOOD"
    elif rms_km < 5.0:
        grade = "⚠️ ACCEPTABLE"
    else:
        grade = "❌ POOR"
    
    print(f"Overall Grade: {grade}")
    print(f"RMS Error: {rms_km:.3f} km")
    print(f"P95 Error: {p95_km:.3f} km")
    
    if pct_1km > 90:
        print("🎯 EXCELLENT: >90% of predictions within 1km of NASA reference")
    elif pct_1km > 70:
        print("👍 GOOD: >70% of predictions within 1km of NASA reference")
    elif pct_1km > 50:
        print("⚠️ MODERATE: >50% of predictions within 1km of NASA reference")
    else:
        print("📝 NEEDS IMPROVEMENT: <50% within 1km of NASA reference")
    
    print(f"\n📝 CONTEXT:")
    print(f"   • This test compares raw SGP4 predictions against NASA's high-precision OEM data")
    print(f"   • The Enhanced EKF system should significantly improve upon these baseline SGP4 results")
    print(f"   • Test period: 12 hours with 10-minute intervals")
    print(f"   • Reference: NASA ISS OEM data from {start_time.strftime('%Y-%m-%d')}")
    
    # Show some example errors
    print(f"\n📋 SAMPLE ERROR DISTRIBUTION:")
    sorted_errors_km = sorted([e/1000 for e in errors])
    print(f"   Smallest errors: {sorted_errors_km[:3]}")
    print(f"   Largest errors:  {sorted_errors_km[-3:]}")

if __name__ == "__main__":
    main()
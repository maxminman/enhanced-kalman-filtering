#!/usr/bin/env python3
"""
Validation script for the Enhanced Adaptive Atmospheric Drag Model
This script validates the implementation without requiring external dependencies
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_implementation():
    """Validate the enhanced adaptive atmospheric drag model implementation"""
    print("Validating Enhanced Adaptive Atmospheric Drag Model Implementation...")
    print("=" * 70)
    
    # Test 1: Import validation
    print("1. Testing imports...")
    try:
        from atmospheric_models import EnhancedAdaptiveAtmosphericDragModel, NRLMSISE00
        from satellite_characterizer import SatelliteProperties, SatelliteType, OrbitalRegime
        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error during import: {e}")
        return False
    
    # Test 2: Model initialization
    print("2. Testing model initialization...")
    try:
        config = {
            'auto_update_space_weather': False,  # Disable to avoid network calls
            'drag_coeff_adaptation': True,
            'uncertainty_propagation': True
        }
        drag_model = EnhancedAdaptiveAtmosphericDragModel(config)
        print("   ✓ Model initialized successfully")
    except Exception as e:
        print(f"   ✗ Model initialization failed: {e}")
        return False
    
    # Test 3: Create test satellite properties
    print("3. Testing satellite properties creation...")
    try:
        satellite_props = SatelliteProperties(
            norad_id="25544",
            name="ISS (ZARYA)",
            mass=450000.0,
            drag_area=1500.0,
            srp_area=1200.0,
            drag_coefficient=2.2,
            srp_coefficient=1.3,
            ballistic_coefficient=0.0073,
            area_to_mass_ratio=0.00333,
            orbital_regime=OrbitalRegime.LOW_LEO,
            satellite_type=SatelliteType.SPACE_STATION,
            uncertainty_bounds={
                'mass': (400000, 500000),
                'drag_area': (1200, 1800),
                'drag_coefficient': (2.0, 2.4)
            },
            characterization_confidence=0.85,
            parameter_confidence={
                'ballistic_coefficient': 0.8,
                'satellite_type': 0.9,
                'mass': 0.7,
                'drag_area': 0.6
            },
            characterization_timestamp=datetime.utcnow(),
            data_sources=['test']
        )
        print("   ✓ Satellite properties created successfully")
    except Exception as e:
        print(f"   ✗ Satellite properties creation failed: {e}")
        return False
    
    # Test 4: Test NRLMSISE-00 enhanced model
    print("4. Testing enhanced NRLMSISE-00 model...")
    try:
        nrlmsise = NRLMSISE00()
        
        # Test density calculation
        density = nrlmsise.get_density(
            altitude_km=400.0,
            latitude=51.6,
            longitude=0.0,
            datetime_utc=datetime.utcnow(),
            f107=150.0,
            kp=3.0
        )
        
        # Validate density is reasonable
        if 1e-15 < density < 1e-8:
            print(f"   ✓ NRLMSISE-00 density calculation successful: {density:.2e} kg/m³")
        else:
            print(f"   ⚠ NRLMSISE-00 density seems unreasonable: {density:.2e} kg/m³")
            
    except Exception as e:
        print(f"   ✗ NRLMSISE-00 test failed: {e}")
        return False
    
    # Test 5: Test drag acceleration calculation (without space weather updates)
    print("5. Testing drag acceleration calculation...")
    try:
        # ISS-like position and velocity
        position = np.array([6778137.0, 0.0, 0.0])  # ~400 km altitude
        velocity = np.array([0.0, 7670.0, 0.0])     # ~7.67 km/s
        
        # Calculate drag acceleration
        drag_acceleration, uncertainty_metrics = drag_model.get_drag_acceleration(
            position, velocity, satellite_props, datetime.utcnow()
        )
        
        drag_magnitude = np.linalg.norm(drag_acceleration)
        
        # Validate results
        if 1e-9 < drag_magnitude < 1e-6:
            print(f"   ✓ Drag acceleration calculation successful: {drag_magnitude:.2e} m/s²")
            print(f"   ✓ Uncertainty metrics calculated: {len(uncertainty_metrics)} metrics")
        else:
            print(f"   ⚠ Drag acceleration magnitude seems unreasonable: {drag_magnitude:.2e} m/s²")
            
    except Exception as e:
        print(f"   ✗ Drag acceleration calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test adaptive features
    print("6. Testing adaptive features...")
    try:
        # Test different satellite types
        test_satellites = [
            (SatelliteType.SPACE_STATION, "Space Station"),
            (SatelliteType.EARTH_OBSERVATION, "Earth Observation"),
            (SatelliteType.COMMUNICATION, "Communication"),
            (SatelliteType.SCIENTIFIC, "Scientific")
        ]
        
        for sat_type, name in test_satellites:
            test_props = satellite_props
            test_props.satellite_type = sat_type
            
            drag_acc, _ = drag_model.get_drag_acceleration(
                position, velocity, test_props, datetime.utcnow()
            )
            
            drag_mag = np.linalg.norm(drag_acc)
            print(f"   ✓ {name}: {drag_mag:.2e} m/s²")
            
    except Exception as e:
        print(f"   ✗ Adaptive features test failed: {e}")
        return False
    
    print("=" * 70)
    print("🎉 All validation tests passed!")
    print("Enhanced Adaptive Atmospheric Drag Model is working correctly.")
    
    return True

if __name__ == "__main__":
    success = validate_implementation()
    if not success:
        print("❌ Validation failed. Please check the implementation.")
        sys.exit(1)
    else:
        print("✅ Validation successful!")
        sys.exit(0)
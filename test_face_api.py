#!/usr/bin/env python3
"""
Test script to isolate the face recognition API error
"""

import sys
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Test imports
    print("Testing imports...")
    from Backend.Survilleance.app import Face_Recognition as face_recognition
    print("✓ Face recognition module imported successfully")
    
    # Test face recognition on a sample image
    test_image_path = "Backend/Survilleance/test_Face_images/face.jpeg"
    known_faces_dir = "Backend/Survilleance/known_faces"
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        sys.exit(1)
        
    if not Path(known_faces_dir).exists():
        print(f"❌ Known faces directory not found: {known_faces_dir}")
        sys.exit(1)
    
    print(f"Testing face recognition with: {test_image_path}")
    
    # Call the face recognition function
    output_path = "test_output.jpg"
    summary = face_recognition.recognize_image(
        test_image_path,
        output_path,
        known_faces_dir
    )
    
    print("✓ Face recognition completed successfully")
    print("Summary type:", type(summary))
    print("Summary content:", summary)
    
    # Test JSON serialization
    print("\nTesting JSON serialization...")
    
    try:
        json_str = json.dumps(summary, indent=2)
        print("✓ Basic JSON serialization works")
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        print("Attempting to fix...")
        
        # Try our custom serializer
        def json_serializer(obj):
            if hasattr(obj, 'dtype'):  # numpy types
                return float(obj) if obj.dtype.kind in 'fc' else int(obj)
            elif hasattr(obj, '__dict__'):  # complex objects
                return str(obj)
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                return str(obj)
        
        try:
            json_str = json.dumps(summary, indent=2, default=json_serializer)
            print("✓ Custom JSON serialization works")
        except Exception as e2:
            print(f"❌ Custom JSON serialization also failed: {e2}")
            
            # Inspect the problematic object
            print("\nDebugging summary structure:")
            for key, value in summary.items():
                print(f"  {key}: {type(value)} = {value}")
                if key == "detections" and isinstance(value, list):
                    for i, det in enumerate(value):
                        print(f"    detection[{i}]: {type(det)}")
                        if isinstance(det, dict):
                            for det_key, det_value in det.items():
                                print(f"      {det_key}: {type(det_value)} = {det_value}")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
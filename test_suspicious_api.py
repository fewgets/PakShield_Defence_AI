import requests
import json
from pathlib import Path

# Test the suspicious activity detection endpoint
API_URL = "http://localhost:8002/border/suspicious/detect"

# Find a test video - check if there's one in the uploads folder
test_video_dir = Path("Backend/BorderAnomly/Suspicious_Activity_Detection_master")
test_videos = list(test_video_dir.glob("*.mp4"))

if not test_videos:
    print("❌ No test video found. Please provide a video file.")
    exit(1)

test_video = test_videos[0]
print(f"📹 Using test video: {test_video}")

print("\n🚀 Sending request to API...")
with open(test_video, 'rb') as f:
    files = {'file': (test_video.name, f, 'video/mp4')}
    response = requests.post(API_URL, files=files)

print(f"\n📊 Response Status: {response.status_code}")
print(f"📊 Response Headers: {dict(response.headers)}")

if response.status_code == 200:
    try:
        data = response.json()
        print("\n✅ API Response JSON:")
        print(json.dumps(data, indent=2))
        
        # Check critical fields
        print("\n🔍 Key Fields Check:")
        print(f"  - status: {data.get('status')}")
        print(f"  - output_url: {data.get('output_url')}")
        print(f"  - video_url: {data.get('video_url')}")
        print(f"  - summary: {data.get('summary')}")
        
        # Try to access the file
        if data.get('output_url'):
            file_url = f"http://localhost:8002{data.get('output_url')}"
            print(f"\n🌐 Testing file access: {file_url}")
            file_response = requests.head(file_url)
            print(f"  - File Status: {file_response.status_code}")
            print(f"  - Content-Type: {file_response.headers.get('content-type')}")
            print(f"  - Content-Length: {file_response.headers.get('content-length')} bytes")
            
            if file_response.status_code == 200:
                print("  ✅ Output file is accessible!")
            else:
                print("  ❌ Output file NOT accessible!")
        
    except Exception as e:
        print(f"\n❌ Error parsing response: {e}")
        print(f"Raw response: {response.text[:500]}")
else:
    print(f"\n❌ Request failed!")
    print(f"Response: {response.text}")

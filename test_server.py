import requests
import json

def test_server():
    base_url = 'http://localhost:5000'
    
    print("🧪 Testing server...")
    
    # Test 1: Health check
    try:
        response = requests.get(f'{base_url}/health')
        if response.status_code == 200:
            print("✅ Health check passed")
            result = response.json()
            print(f"   Model loaded: {result['model_loaded']}")
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Server not running: {e}")
        return
    
    # Test 2: Image classification
    try:
        with open('test_images/plastic_test.jpg', 'rb') as f:
            files = {'image': f}
            response = requests.post(f'{base_url}/classify', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Classification test passed")
            print(f"   Predicted: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.3f}")
        else:
            print("❌ Classification test failed")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Classification test error: {e}")
    
    # Test 3: Statistics
    try:
        response = requests.get(f'{base_url}/statistics')
        if response.status_code == 200:
            print("✅ Statistics test passed")
            result = response.json()
            print(f"   Total classifications: {result['all_time']['total']}")
        else:
            print("❌ Statistics test failed")
    except Exception as e:
        print(f"❌ Statistics test error: {e}")

if __name__ == '__main__':
    test_server()
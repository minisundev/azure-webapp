import requests
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

SUBSCRIPTION_KEY = os.getenv("SUBSCRIPTION_KEY2")
ENDPOINT = os.getenv("ENDPOINT2")

# 이미지 분석
def analyze_image(image_url):
    analyze_url = ENDPOINT + "vision/v3.2/analyze"
    
    headers = {
        'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
        'Content-Type': 'application/json'
    }
    
    params = {
        'visualFeatures': 'Categories,Description,Color',
        'language': 'en'
    }
    
    data = {
        'url': image_url
    }
    
    response = requests.post(analyze_url, headers=headers, params=params, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Object Detection
def detect_objects(image_url):
    detect_url = ENDPOINT + "vision/v3.2/detect"
    
    headers = {
        'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
        'Content-Type': 'application/json'
    }
    
    # Object Detection은 params 불필요!
    
    data = {
        'url': image_url
    }
    
    # params 제거
    response = requests.post(detect_url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def main():
    image_url = input("Enter the image URL: ")
    
    choice = input("Choose analysis type (1: Analyze Image, 2: Detect Objects): ")
    
    try:
        if choice == '1':
            result = analyze_image(image_url)
            print("\n=== Analysis Result ===")
            print(result)
            
            # 주요 정보 추출 (원래 코드 방식 유지)
            print("Description:", result.get('description', {}).get('captions', [{}])[0].get('text', 'No description available'))
            
            if 'description' in result and result['description']['captions']:
                print(f"Confidence: {result['description']['captions'][0]['confidence']:.2f}")
            
            if 'categories' in result:
                print(f"Categories: {[cat['name'] for cat in result['categories']]}")
                
        elif choice == '2':
            result = detect_objects(image_url)
            print("\n=== Object Detection Result ===")
            print(result)
            
            # 감지된 객체들 출력 (안전한 방식)
            objects = result.get('objects', [])
            if objects:
                print(f"Found {len(objects)} objects:")
                for i, obj in enumerate(objects, 1):
                    print(f"{i}. Object: {obj.get('object', 'Unknown')}")
                    print(f"   Confidence: {obj.get('confidence', 0):.2f}")
                    print(f"   Bounding Box: {obj.get('rectangle', {})}")
            else:
                print("Objects:", result.get('objects', 'No objects detected'))
                
        else:
            print("Invalid choice. Please select 1 or 2.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
import requests
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from urllib.parse import urlparse
from io import BytesIO

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
    
    data = {
        'url': image_url
    }
    
    response = requests.post(detect_url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Create bounding box
def create_bounding_box(image_url, detection_result):
    try:
        # URL에서 이미지 다운로드
        response = requests.get(image_url)
        response.raise_for_status()  # HTTP 에러 확인
        
        # BytesIO를 사용해서 메모리에서 이미지 열기
        image = Image.open(BytesIO(response.content))
        
        # image 폴더 생성
        os.makedirs("image", exist_ok=True)
        
        # URL에서 파일명 추출
        parsed_url = urlparse(image_url)
        original_filename = os.path.basename(parsed_url.path)
        
        if '.' in original_filename:
            name, ext = original_filename.rsplit('.', 1)
        else:
            name = "downloaded_image"
            ext = "jpg"
        
        # 원본 이미지 저장
        original_path = os.path.join("image", f"{name}.{ext}")
        image.save(original_path)
        print(f"Original image saved at: {original_path}")
        
    except Exception as e:
        raise Exception(f"Error downloading/opening image: {e}")
    
    # Bounding box 그리기
    draw = ImageDraw.Draw(image)
    
    # 폰트 사이즈 설정 (큰 폰트)
    try:
        # macOS
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 100)
    except:
        # Windows
        try:
            font = ImageFont.truetype("arial.ttf", 100)
        except:
            font = ImageFont.load_default()
    
    for obj in detection_result.get('objects', []):
        rectangle = obj.get('rectangle')
        if rectangle:
            x, y, w, h = rectangle.get('x', 0), rectangle.get('y', 0), rectangle.get('w', 0), rectangle.get('h', 0)
            # Draw bounding box
            draw.rectangle([x, y, x + w, y + h], outline='red', width=2)  # 선 두께 다시 얇게
            # Draw label with much larger font
            draw.text((x, y), obj.get('object', 'Unknown'), fill='red', font=font)
    
    # Bounding box가 그려진 이미지 저장
    bbox_filename = f"{name}_with_bounding_box.{ext}"
    bbox_path = os.path.join("image", bbox_filename)
    
    image.save(bbox_path)
    print(f"Image with bounding boxes saved at: {bbox_path}")
    image.show()  # Show the image with bounding boxes


def main():
    image_url = input("Enter the image URL: ")
    
    choice = input("Choose analysis type (1: Analyze Image, 2: Detect Objects, 3: Bounding box): ")
    
    try:
        if choice == '1':
            result = analyze_image(image_url)
            print("\n=== Analysis Result ===")
            print(result)
            
            # 주요 정보 추출
            print("Description:", result.get('description', {}).get('captions', [{}])[0].get('text', 'No description available'))
            
            if 'description' in result and result['description']['captions']:
                print(f"Confidence: {result['description']['captions'][0]['confidence']:.2f}")
            
            if 'categories' in result:
                print(f"Categories: {[cat['name'] for cat in result['categories']]}")
                
        elif choice == '2':
            result = detect_objects(image_url)
            print("\n=== Object Detection Result ===")
            print(result)
            
            # 감지된 객체들 출력
            objects = result.get('objects', [])
            if objects:
                print(f"Found {len(objects)} objects:")
                for i, obj in enumerate(objects, 1):
                    print(f"{i}. Object: {obj.get('object', 'Unknown')}")
                    print(f"   Confidence: {obj.get('confidence', 0):.2f}")
                    print(f"   Bounding Box: {obj.get('rectangle', {})}")
            else:
                print("Objects:", result.get('objects', 'No objects detected'))
                
        elif choice == '3':
            # 먼저 객체 감지 수행
            detection_result = detect_objects(image_url)
            print("\n=== Object Detection Result ===")
            print(detection_result)
            
            # Create bounding box
            create_bounding_box(image_url, detection_result)
                
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
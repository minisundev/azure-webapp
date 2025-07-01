import requests
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

SUBSCRIPTION_KEY = os.getenv("SUBSCRIPTION_KEY")
ENDPOINT = os.getenv("ENDPOINT")

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
    
def main():
    image_url = "https://variety.com/wp-content/uploads/2025/06/Kpop-Demon-Hunters-.jpg?crop=170px%2C0px%2C1200px%2C800px&resize=1200%2C800"
    
    try:
        analysis = analyze_image(image_url)
        print("Analysis Result:")
        print(analysis)
    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()

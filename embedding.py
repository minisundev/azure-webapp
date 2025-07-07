import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI  # 최신 SDK 사용
from tqdm import tqdm

# 1️⃣ .env 로드 + OpenAI 설정
print("📦 Loading .env...")
load_dotenv()

# 환경변수 확인
print("🔍 Checking environment variables...")
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
api_type = os.getenv("OPENAI_API_TYPE", "azure")
api_version = os.getenv("OPENAI_API_VERSION", "2023-05-15")
deployment_id = os.getenv("DEPLOYMENT_ID")

print(f"API Key: {'✅ Found' if api_key else '❌ Missing'}")
print(f"API Base: {api_base}")
print(f"API Type: {api_type}")
print(f"API Version: {api_version}")
print(f"Deployment ID: {deployment_id}")

# OpenAI 클라이언트 설정 (최신 SDK 방식)
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=api_base
)

print(f"🔧 Client configured with endpoint: {api_base}")
print(f"🔧 API Version: {api_version}")
print(f"🔧 Deployment: {deployment_id}")

DEPLOYMENT_ID = deployment_id

# 2️⃣ 파일 경로
INPUT_PATH = "./data/All_Beauty_5.json"
OUTPUT_PATH = "./data/All_Beauty_5_embedded.json"

print(f"\n📂 File paths:")
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")

# 3️⃣ 파일 읽기 (디버깅 추가)
def load_json_lines(filepath):
    print(f"\n🔍 Loading JSON lines from: {filepath}")
    
    # 파일 존재 확인
    if not os.path.exists(filepath):
        print(f"❌ File does not exist: {filepath}")
        return []
    
    # 파일 크기 확인
    file_size = os.path.getsize(filepath)
    print(f"📊 File size: {file_size:,} bytes")
    
    data = []
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            print("📖 Reading file line by line...")
            for line_num, line in enumerate(f):
                try:
                    line = line.strip()
                    if not line:  # 빈 줄 스킵
                        continue
                    
                    item = json.loads(line)
                    data.append(item)
                    
                    # 첫 3개 아이템 샘플 출력
                    if line_num < 3:
                        print(f"  Sample {line_num}: {str(item)[:100]}...")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Unexpected error at line {line_num}: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []
    
    print(f"✅ Successfully loaded {len(data)} items")
    return data

def save_json_lines(filepath, data):
    print(f"\n💾 Saving {len(data)} items to: {filepath}")
    
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w", encoding='utf-8') as f:
            for i, item in enumerate(data):
                try:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    if i % 100 == 0:  # 100개마다 진행상황 출력
                        print(f"  💾 Saved {i+1}/{len(data)} items")
                except Exception as e:
                    print(f"❌ Error saving item {i}: {e}")
                    continue
        
        print(f"✅ Successfully saved {len(data)} items to {filepath}")
        
        # 저장된 파일 크기 확인
        saved_size = os.path.getsize(filepath)
        print(f"📊 Saved file size: {saved_size:,} bytes")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")

# 4️⃣ 임베딩 함수 (최신 SDK 방식)
def get_embedding(text):
    print(f"    🔄 Calling embedding API...")
    print(f"    📝 Text length: {len(text)} characters")
    print(f"    🎯 Using deployment: {DEPLOYMENT_ID}")
    
    try:
        response = client.embeddings.create(
            input=text,
            model=DEPLOYMENT_ID  # 최신 SDK에서는 model 파라미터 사용
        )
        
        embedding = response.data[0].embedding
        print(f"    ✅ Got embedding with {len(embedding)} dimensions")
        return embedding
        
    except Exception as e:
        print(f"    ❌ Embedding API error: {e}")
        print(f"    🔍 Error type: {type(e)}")
        raise

# 5️⃣ 메인 실행 (디버깅 추가)
if __name__ == "__main__":
    print("🚀 Starting embedding process...")
    
    # 입력 데이터 로드
    print("\n" + "="*50)
    print("STEP 1: Loading input data")
    print("="*50)
    
    data = load_json_lines(INPUT_PATH)
    
    if not data:
        print("❌ No data loaded. Exiting.")
        exit(1)
    
    print(f"🔍 Total items loaded: {len(data)}")
    
    # 첫 번째 아이템 구조 확인
    if data:
        first_item = data[0]
        print(f"\n📋 First item structure:")
        for key, value in first_item.items():
            print(f"  {key}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
    
    enriched = []
    
    print("\n" + "="*50)
    print("STEP 2: Processing embeddings")
    print("="*50)
    
    for i, item in enumerate(tqdm(data, desc="Processing items")):
        try:
            print(f"\n[{i+1}/{len(data)}] Processing item {i}...")
            
            # reviewText 확인
            text = item.get("reviewText", "")
            if not text:
                print(f"[{i}] ⚠️ Empty reviewText, skipping")
                continue
            
            # 텍스트 길이 제한 (8000자 이상이면 자르기)
            original_length = len(text)
            if len(text) > 8000:
                text = text[:8000]
                print(f"[{i}] ✂️ Text truncated from {original_length} to {len(text)} chars")
            
            print(f"[{i}] 📝 Text preview: {text[:80]}...")
            
            # 임베딩 생성
            vector = get_embedding(text)
            
            # 결과 저장
            item["id"] = str(i)
            item["embedding"] = vector
            enriched.append(item)
            
            print(f"[{i}] ✅ Success — vector length: {len(vector)}")
            
            # 10개마다 중간 저장
            if (i + 1) % 10 == 0:
                temp_path = OUTPUT_PATH.replace(".json", f"_temp_{i+1}.json")
                save_json_lines(temp_path, enriched)
                print(f"📦 Temporary backup saved: {temp_path}")
            
        except Exception as e:
            print(f"[{i}] ❌ Error processing item: {e}")
            print(f"[{i}] 🔍 Error type: {type(e)}")
            continue
    
    print("\n" + "="*50)
    print("STEP 3: Saving final results")
    print("="*50)
    
    if enriched:
        save_json_lines(OUTPUT_PATH, enriched)
        print(f"✅ Done! Total embedded: {len(enriched)}/{len(data)}")
        print(f"📊 Success rate: {len(enriched)/len(data)*100:.1f}%")
    else:
        print("❌ No items were successfully processed!")
    
    print("\n🎉 Process completed!")
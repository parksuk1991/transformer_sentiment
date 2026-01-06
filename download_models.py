# download_models.py
import os
import ssl
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# SSL 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(InsecureRequestWarning)

# 환경 변수 설정
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 50)
print("FinBERT 모델 다운로드 시작...")
print("=" * 50)

try:
    model_name = "ProsusAI/finbert"
    save_path = "./models/finbert"
    
    # 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n1️⃣  Tokenizer 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_path)
    print(f"✅ Tokenizer 저장 완료: {save_path}")
    
    print(f"\n2️⃣  Model 다운로드 중... (크기: ~400MB, 시간 소요)")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model.save_pretrained(save_path)
    print(f"✅ Model 저장 완료: {save_path}")
    
    print("\n" + "=" * 50)
    print("✅ 모든 모델 다운로드 완료!")
    print(f"저장 경로: {os.path.abspath(save_path)}")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    print("\n💡 해결책:")
    print("1. 인터넷 연결 확인")
    print("2. 방화벽 설정 확인")
    print("3. 이메일/전화로 IT 부서에 문의")

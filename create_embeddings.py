import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Khởi tạo mô hình tạo embeddings (sử dụng model đa ngôn ngữ)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Lấy dữ liệu từ API
def fetch_data():
    url = "https://ef014c9dfc4b.ngrok-free.app/api/v1/ai"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi lấy dữ liệu từ API: {e}")
        return None

# Tạo embeddings cho dữ liệu
def create_embeddings(data):
    # Lấy danh sách các text từ trường 'text' trong mỗi document
    texts = [doc.get('text', '') for doc in data.get('result', {}).get('documents', [])]
    print(f"Found {len(texts)} documents to embed")
    
    # In thử 1-2 text đầu tiên để kiểm tra
    # for i, text in enumerate(texts[:2], 1):
    #     print(f"\nText {i} (first 100 chars): {text[:100]}...")
    
    # Tạo embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

# Hàm chính
def main():
    # Lấy dữ liệu từ API
    print("Fetching data from API...")
    data = fetch_data()
    if not data:
        print("Không thể lấy dữ liệu từ API")
        return
    
    print("\nData structure received:")
    print(f"Total documents: {data.get('result', {}).get('total_documents', 0)}")
    # Tạo embeddings
    print("\nCreating embeddings...")
    embeddings = create_embeddings(data)
    
    # Lưu embeddings vào file
    np.save('embeddings.npy', embeddings)
    print(f"\nĐã tạo và lưu embeddings cho {len(embeddings)} documents")
    print(f"Kích thước embeddings: {embeddings.shape}")
    print(f"Độ dài mỗi vector embedding: {embeddings.shape[1] if len(embeddings.shape) > 1 else 'N/A'}")
    
    # Lưu metadata
    if 'result' in data and 'documents' in data['result']:
        metadata = {
            "documents": data['result']['documents'],
            "total_documents": data.get('result', {}).get('total_documents', 0)
        }
        with open('metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print("Đã lưu metadata vào file metadata.json")
        
if __name__ == "__main__":
    main()
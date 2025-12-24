from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from pyngrok import ngrok
import nest_asyncio
import uvicorn
import threading
app = FastAPI()

# Load model và dữ liệu đã lưu
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = np.load('embeddings.npy')
with open('metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
def start_ngrok():
    ngrok.set_auth_token("366ygBRWgfxKlzNBxZzXefOjnvH_atKtNUCPJumGkHxeZCyj")  # Thay thế bằng auth token của bạn
    ngrok_tunnel = ngrok.connect(8000)
    print(f"Public URL: {ngrok_tunnel.public_url}")
class AiRequest(BaseModel):
    message: str
    uIdFE: str

def find_most_similar(query_embedding: np.ndarray, user_id: str = None, max_results: int = 5) -> List[str]:
    # Tính độ tương đồng giữa câu hỏi và tất cả văn bản
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]
    
    # Sắp xếp các chỉ số theo độ tương đồng giảm dần
    sorted_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in sorted_indices:
        doc = metadata['documents'][idx]
        
        # Lọc theo user_id nếu có
        if user_id and str(doc.get('metadata', {}).get('user_id')) != user_id:
            continue
            
        # Thêm nội dung văn bản vào kết quả
        results.append(doc['text'])
        
        # Dừng khi đủ số lượng kết quả cần lấy
        if len(results) >= max_results:
            break
    
    return results

def generate_response(user_message: str, context: List[str]) -> str:
    context_text = "\n".join([f"- {item}" for item in context])
    prompt = f"""Based on the following transaction information, please answer the user's question concisely and clearly.
Transaction Information:
{context_text}
Question {user_message}
Answer:"""
    response = ollama.chat(
        model='gemma:2b',  # hoặc model khác bạn muốn dùng
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

@app.post("/api/chat")
async def chat(request: AiRequest):
    try:
        # Tạo embedding cho câu hỏi
        query_embedding = model.encode(
            [request.message], 
            convert_to_numpy=True,
            show_progress_bar=False  # Tắt thanh tiến trình để đỡ tốn tài nguyên
        )[0]
        
        # Lấy các kết quả tương đồng (tối đa 5 kết quả)
        context = find_most_similar(
            query_embedding=query_embedding,
            user_id=request.uIdFE,
            max_results=5  # Chỉ lấy 5 kết quả có độ tương đồng cao nhất
        )
        
        # Tạo câu trả lời nếu có kết quả
        if not context:
            return {
                "userId": request.uIdFE,
                "answer": "Không tìm thấy thông tin phù hợp."
            }
            
        # Tạo câu trả lời tự nhiên
        answer = generate_response(request.message, context)
        
        return {
            "userId": request.uIdFE,
            "answer": answer
        }
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")  # In lỗi ra console để debug
        raise HTTPException(
            status_code=500,
            detail=f"Có lỗi xảy ra: {str(e)}"
        )
    
if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
        # Khởi động ngrok trong một luồng riêng biệt
    threading.Thread(target=start_ngrok).start()
    
    # Cho phép chạy nhiều event loop
    nest_asyncio.apply()
    
    # Khởi động FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import tempfile
import os
import requests
from pyngrok import ngrok
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
# Khởi tạo app
app = Flask(__name__)
CORS(app)
ngrok_url = ""
class InvoiceProcessor:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate"):
        """
        Initialize the Invoice Processor
        
        Args:
            ollama_url: URL of the Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self.model = "llama3" 
        self.api_base_url = os.getenv("API_BASE_URL")
        self.categories = self._fetch_categories()

    # Lấy danh sách danh mục từ API
    def _fetch_categories(self) -> List[Dict[str, str]]:
        """Lấy danh sách danh mục từ API và in ra tên các danh mục"""
        try:
            url = f"{self.api_base_url}/categories"
            print("Đang tải danh sách danh mục...")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Lấy danh sách category từ data['result']
            categories = [{'name': item['categoryname']} for item in data['result']]
            
            print("\nDanh sách danh mục:")
            for idx, category in enumerate(categories, 1):
                print(f"{idx}. {category['name']}")
                
            return categories
                
        except Exception as e:
            print(f"Lỗi khi lấy danh mục: {str(e)}")
            if 'response' in locals():
                print(f"Response content: {response.text[:1000]}")
            return []

    # Gọi API Ollama
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with the given prompt"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1
                }
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"⚠️ Lỗi khi gọi Ollama: {str(e)}")
            return ""
    
    # Phân loại hóa đơn
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using EasyOCR"""
        try:
            reader = easyocr.Reader(['en'])
            #Đọc văn bản từ ảnh
            results = reader.readtext(image_path)
            #Kết hợp tất cả các dòng văn bản
            return ' '.join([result[1] for result in results])
        except Exception as e:
            print(f"⚠️ Lỗi khi nhận dạng văn bản: {str(e)}")
            return ""

    # Phân loại hóa đơn
    def classify_invoice(self, invoice_text: str) -> Dict[str, str]:
        """Classify invoice and extract total amount"""
        if not self.categories:
            return {"category": "Uncategorized", "total": "0"}
            
        category_names = [cat['name'] for cat in self.categories]
        categories_list = '\n'.join(f'- {name}' for name in category_names)
        
        prompt = f"""You are an invoice analysis expert. Please:

    1. Classify the following invoice into one of these categories (respond with category name only):
    {categories_list}

    2. Find and extract the total amount from the invoice content (respond with numbers only, no special characters or currency units)

    Invoice content:
    {invoice_text[:2000]}

    Respond in the following JSON format (ONLY respond with JSON, no other text):
    {{
        "category": "category_name",
        "total": "amount"
    }}"""
        
        try:
            response = self._call_ollama(prompt)
            # Parse the JSON response
            import json
            result = json.loads(response.strip())
            return {
                "category": result.get("category", "Unknown"),
                "total": result.get("total", "0")
            }
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return {"category": "Parse Error", "total": "0"}

    #Xử lý hóa đơn
    def process_invoice(self, image_path: str) -> Dict[str, str]:
        """
        Process an invoice image and return the classification result
        
        Returns:
            Dictionary containing:
            - status: "success" or "error"
            - text: Extracted text (if successful)
            - category: Classified category (if successful)
            - total: Extracted total amount (if found)
            - error: Error message (if any)
        """
        if not Path(image_path).exists():
            return {"status": "error", "error": "File does not exist"}
            
        print("Extracting text from image...")
        text = self.extract_text(image_path)
        if not text:
            return {"status": "error", "error": "Failed to extract text"}
            
        print("Analyzing invoice...")
        result = self.classify_invoice(text)
        
        return {
            "status": "success",
            "text": text,
            "category": result["category"],
            "total": result["total"]
        }
   
    # API endpoint để xử lý hóa đơn từ URL

@app.route('/process_invoice', methods=['POST'])
def process_invoice_api():
    """API endpoint để xử lý hóa đơn từ URL"""
    try:
        data = request.json
        if not data or 'urls' not in data or not isinstance(data['urls'], list):
            return jsonify({
                "status": "error",
                "error": "Thiếu tham số urls hoặc không đúng định dạng (phải là mảng)"
            }), 400

        results = []
        processor = InvoiceProcessor()
        
        for url in data['urls']:
            try:
                # Tải ảnh từ URL
                response = requests.get(url)
                if response.status_code != 200:
                    results.append({
                        "url": url,
                        "status": "error",
                        "error": f"Không thể tải ảnh. Mã lỗi: {response.status_code}"
                    })
                    continue

                # Lưu ảnh vào file tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name

                try:
                    # Xử lý ảnh
                    result = processor.process_invoice(temp_path)
                    
                    invoice_result = {
                        "invoice_type": result.get("category", "Không xác định"),
                        "total_amount": result.get("total", "0")
                    }
                    results.append(invoice_result)
                    
                    # In kết quả ra console
                    print("\n" + "="*50)
                    print(f"URL: {url}")
                    print("-"*50)
                    print(f"invoice_type: {result.get('category', 'Không xác định')}")
                    print(f"total_amount: {result.get('total', '0')}")
                    print("="*50 + "\n")
                    
                except Exception as e:
                    results.append({
                        "url": url,
                        "status": "error",
                        "error": f"Lỗi khi xử lý ảnh: {str(e)}"
                    })
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

            except Exception as e:
                results.append({
                    "url": url,
                    "status": "error",
                    "error": f"Lỗi không xác định: {str(e)}"
                })

        # Trả về kết quả đầu tiên nếu có
        if results:
            return jsonify(results[0])
        return jsonify({
            "status": "error",
            "error": "Không có kết quả nào được xử lý"
        })
        
    except Exception as e:
        print(f"Lỗi hệ thống: {str(e)}")
        return jsonify({
            "status": "error",
            "error": f"Lỗi hệ thống: {str(e)}"
        }), 500
# Load model và dữ liệu
model = SentenceTransformer('keepitreal/vietnamese-sbert')
# Đọc dữ liệu từ metadata.json
with open('metadata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print("Type of data:", type(data))
    if isinstance(data, list):
        print("First item type:", type(data[0]) if data else "Empty list")
        print("First item keys:", data[0].keys() if data and isinstance(data[0], dict) else "Not a dictionary")
    else:
        print("Data is not a list")
# Tách embeddings và documents
embeddings = np.array([item['embedding'] for item in data])
documents = [{'text': item['text'], 'metadata': item['metadata']} for item in data]
class AiRequest(BaseModel):
    uIdFE: str
    message: str
def find_most_similar(query_embedding: np.ndarray, user_id: str = None, max_results: int = 5) -> List[str]:
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]
    sorted_indices = np.argsort(similarities)[::-1]
    results = []
    for idx in sorted_indices:
        doc = documents[idx]
        if user_id and str(doc.get('metadata', {}).get('user_id')) != user_id:
            continue
        results.append(doc['text'])
        if len(results) >= max_results:
            break
    return results
def generate_response(user_message: str, context: List[str]) -> str:
    context_text = "\n".join([f"- {item}" for item in context])
    prompt = f"""Based on the following transaction information, please answer the user's question concisely and clearly.
Transaction Information:
{context_text}
Question: {user_message}
Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json().get("response", "").strip()
@app.post("/api/chat")
async def chat():
    try:
        request_data = request.get_json()
        if not request_data or 'uIdFE' not in request_data or 'message' not in request_data:
            return jsonify({
                "userId": request_data.get('uIdFE', 'unknown'),
                "answer": "Thiếu thông tin uIdFE hoặc message"
            }), 400
        query_embedding = model.encode(
            [request_data['message']], 
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]
        
        context = find_most_similar(
            query_embedding=query_embedding,
            user_id=request_data['uIdFE'],
            max_results=5
        )
        
        if not context:
            return jsonify({
                "userId": request_data['uIdFE'],
                "answer": "Không tìm thấy giao dịch nào phù hợp."
            })
            
        answer = generate_response(request_data['message'], context)
        return jsonify({
            "userId": request_data['uIdFE'],
            "answer": answer
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "userId": request_data.get('uIdFE', 'unknown'),
            "answer": f"Có lỗi xảy ra: {str(e)}"
        }), 500
def start_ngrok():
    global ngrok_url
    try:
        ngrok.kill()  # Tắt kết nối cũ nếu có
        ngrok_tunnel = ngrok.connect(5000)
        ngrok_url = ngrok_tunnel.public_url
        print(f" * Đường dẫn ngrok: {ngrok_url}")
    except Exception as e:
        print(f"Lỗi khi khởi động ngrok: {str(e)}")
        ngrok_url = "Không thể khởi động ngrok"

if __name__ == "__main__":
    # Khởi động ngrok
    start_ngrok()
    
    # Khởi động server
    print(" * Đang khởi động máy chủ...")
    app.run(port=5000)
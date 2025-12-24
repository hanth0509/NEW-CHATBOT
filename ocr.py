import json
import os
import requests
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import easyocr
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import tempfile
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
CORS(app)  # Cho phép gọi API từ các domain khác
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
                    processor = InvoiceProcessor()
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
        if results:
            return jsonify(results[0])
        return jsonify({})
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Lỗi hệ thống: {str(e)}"
        }), 500
    
def start_ngrok():
        """Khởi động ngrok tunnel và trả về URL công khai"""
        global ngrok_url
        # Tắt các kết nối cũ nếu có
        ngrok.kill()
        # Tạo tunnel mới
        ngrok_tunnel = ngrok.connect(5000)
        ngrok_url = ngrok_tunnel.public_url
        print(f" * Đường dẫn ngrok: {ngrok_url}")

if __name__ == "__main__":
    # Initialize the processor
    # processor = InvoiceProcessor()
    
    # Process an invoice
    # result = processor.process_invoice("hd.jpg")

    # Print results
    # print("\n" + "="*50)
    # if result["status"] == "success":
    #     print(" Processing successful!")
    #     print(f"\nExtracted text (first 500 chars):\n{result['text'][:500]}...")
    #     print(f"\nCategory: {result['category']}")
    #     print(f"Total amount: {result['total']}")
    # else:
    #     print(f"Error: {result.get('error', 'Unknown error')}")
    
    start_ngrok()
    # Chạy ứng dụng Flask
    print(" * Đang khởi động máy chủ Flask...")
    app.run(port=5000)
#LOGIC HÓA ĐƠN
import os
import requests
import tempfile
import easyocr
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
import json
import requests
load_dotenv()

class InvoiceProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(["en"])
        self.model = "llama3"
        self.ollama_url = "http://localhost:11434/api/generate"
        self.api_base_url = os.getenv("API_BASE_URL")
        self.categories = self._fetch_categories()

    def _fetch_categories(self) -> List[Dict[str, str]]:
        try:
            res = requests.get(
                f"{self.api_base_url}/categories",
                timeout=10
            )
            res.raise_for_status()
            return [{"name": c["categoryname"]} for c in res.json()["result"]]
        except:
            return []

    def _call_ollama(self, prompt: str) -> str:
        try:
            # Sử dụng ollama python client thay vì requests
            import ollama
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ Lỗi khi gọi Ollama: {str(e)}")
            raise   

    def extract_text(self, image_path: str) -> str:
        result = self.reader.readtext(image_path)
        return " ".join([r[1] for r in result])

    def classify_invoice(self, text: str) -> Dict[str, str]:
        print("Debug - Categories:", self.categories)  
        if not self.categories:
            return {"category": "Uncategorized", "total": "0"}

        category_names = [c["name"] for c in self.categories]
        categories_list = "\n".join(f"- {name}" for name in category_names)

        prompt = f"""
    You are an invoice analysis expert specialized in Vietnamese invoices.

    Your tasks:
    1. Classify the invoice into ONE category from the list below.
    - Respond with the category name only.
    - Choose the most relevant category.

    2. Extract the FINAL TOTAL amount that the customer has to pay.
    - This is usually labeled in Vietnamese as:
        "Tổng cộng", "Tổng tiền", "Thành tiền", "Tổng thanh toán",
        "Cộng tiền", "Số tiền phải trả".
    - If multiple amounts appear, choose the LARGEST and FINAL amount.
    - Ignore VAT-only amounts, unit prices, discounts, and cash given by customer.
    - Respond with numbers only (digits).
    - Do NOT include currency symbols like VND, ₫, đ.
    - Do NOT include separators like commas or dots.

    Categories:
    {categories_list}

    Invoice content (Vietnamese OCR text):
    {text[:2000]}

    IMPORTANT RULES:
    - Respond ONLY with valid JSON
    - Do NOT explain your reasoning
    - Do NOT add any text outside the JSON
    - Do NOT use markdown

    JSON format:
    {{
    "category": "category_name",
    "total": "amount"
    }}
    """


        try:
            result = self._call_ollama(prompt).strip()
            data = json.loads(result)
            return {
                "category": data.get("category", "Unknown"),
                "total": data.get("total", "0")
            }
        except requests.exceptions.Timeout:
            print("❌ Ollama API request timed out")
            return {"category": "Unknown", "total": "0"}
        except Exception as e:
            print("❌ Error processing invoice:", e)
            if 'result' in locals():
                print("Raw response:", result)
            return {"category": "Unknown", "total": "0"}


    def process_invoice_from_url(self, url: str) -> Dict[str, str]:
        res = requests.get(url) 
        if res.status_code != 200:
            return {"error": "Download failed"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(res.content)
            path = f.name

        try:
            text = self.extract_text(path)
            result = self.classify_invoice(text)
            # Return the dictionary directly without putting it in a list
            return {
                "invoice_type": result["category"],
                "total_amount": result["total"]
            }
        finally:
            if Path(path).exists():
                os.unlink(path)

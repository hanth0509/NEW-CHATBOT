#LOGIC HÓA ĐƠN
import os
import requests
import tempfile
import easyocr
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
import json

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
        res = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
            }
        )
        return res.json().get("response", "")

    def extract_text(self, image_path: str) -> str:
        result = self.reader.readtext(image_path)
        return " ".join([r[1] for r in result])

    def classify_invoice(self, text: str) -> Dict[str, str]:
        names = "\n".join(f"- {c['name']}" for c in self.categories)

        prompt = f"""
Classify invoice and extract total.

Categories:
{names}

Invoice:
{text[:2000]}

Respond JSON only:
{{"category":"...", "total":"..."}}
"""

        try:
            result = self._call_ollama(prompt)
            data = json.loads(result)
            return {
            "category": data.get("category", "Unknown"),
            "total": data.get("total", "0")
        }
        except:
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
            return {
                "invoice_type": result["category"],
                "total_amount": result["total"]
            }
        finally:
            if Path(path).exists():
                os.unlink(path)

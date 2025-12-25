from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import threading
import nest_asyncio
from pyngrok import ngrok

from services.rag_service import model, find_most_similar, generate_response
from services.invoice_service import InvoiceProcessor

app = FastAPI()
invoice_processor = InvoiceProcessor()

# ========== SCHEMA ==========
class ChatRequest(BaseModel):
    message: str
    uIdFE: str

class InvoiceRequest(BaseModel):
    urls: List[str]

# ========== API CHAT ==========
@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        emb = model.encode([req.message], convert_to_numpy=True)[0]
        context = find_most_similar(emb, req.uIdFE)

        if not context:
            return {"answer": "Không tìm thấy dữ liệu phù hợp"}

        answer = generate_response(req.message, context)
        return {"userId": req.uIdFE, "answer": answer}

    except Exception as e:
        raise HTTPException(500, str(e))

# ========== API INVOICE ==========
@app.post("/process_invoice")
def process_invoice(req: InvoiceRequest):
    if not req.urls:
        return {"error": "No URLs provided"}
    
    # If only one URL, return a single object instead of a list
    if len(req.urls) == 1:
        try:
            return invoice_processor.process_invoice_from_url(req.urls[0])
        except Exception as e:
            return {"error": str(e), "url": req.urls[0]}
    
    # If multiple URLs, return a list of results
    results = []
    for url in req.urls:
        try:
            result = invoice_processor.process_invoice_from_url(url)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "url": url
            })
    return results

# ========== MAIN ==========
def start_ngrok():
    ngrok.set_auth_token("366ygBRWgfxKlzNBxZzXefOjnvH_atKtNUCPJumGkHxeZCyj")
    public_url = ngrok.connect(8000).public_url
    print("🌍 Public URL:", public_url)

if __name__ == "__main__":
    threading.Thread(target=start_ngrok).start()
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

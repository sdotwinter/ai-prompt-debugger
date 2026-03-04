from fastapi import FastAPI
from pydantic import BaseModel
import tiktoken

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    model_name: str = "gpt-3.5-turbo"

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: PromptRequest):
    enc = tiktoken.encoding_for_model(req.model_name)
    tokens = enc.encode(req.prompt)
    return {
        "original_prompt": req.prompt,
        "token_count": len(tokens),
        "tokens": tokens[:50],
        "max_context": 4096 if "gpt-4" not in req.model_name else 8192
    }

@app.post("/tokenize")
def tokenize(req: PromptRequest):
    enc = tiktoken.encoding_for_model(req.model_name)
    tokens = enc.encode(req.prompt)
    return {"token_count": len(tokens), "tokens": tokens}

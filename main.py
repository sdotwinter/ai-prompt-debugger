from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import re
import json
import tiktoken

app = FastAPI(
    title="AI Prompt Debugger",
    description="A tool for analyzing and debugging AI prompts",
    version="1.0.0"
)

class PromptRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"

class AnalysisResponse(BaseModel):
    original_prompt: str
    token_count: int
    word_count: int
    character_count: int
    potential_issues: List[str]
    suggestions: List[str]
    complexity_score: float
    readability_score: float

class TokenizeResponse(BaseModel):
    tokens: List[str]
    token_ids: List[int]
    count: int

@app.get("/")
async def root():
    return {"message": "AI Prompt Debugger API"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_prompt(request: PromptRequest):
    """
    Analyze a prompt for potential issues and provide suggestions
    """
    prompt = request.prompt
    
    # Count tokens, words, and characters
    enc = tiktoken.encoding_for_model(request.model)
    token_ids = enc.encode(prompt)
    token_count = len(token_ids)
    word_count = len(prompt.split())
    character_count = len(prompt)
    
    # Identify potential issues
    issues = []
    suggestions = []
    
    # Check for very short prompts
    if word_count < 5:
        issues.append("Prompt might be too short - consider adding more context")
        suggestions.append("Add more specific details about what you want the AI to do")
    
    # Check for very long prompts
    if token_count > 2000:
        issues.append("Prompt is very long and might exceed context limits")
        suggestions.append("Consider breaking down the prompt into smaller parts")
    
    # Check for ambiguous language
    ambiguous_words = ['it', 'this', 'that', 'these', 'those']
    found_ambiguous = [word for word in ambiguous_words if f' {word} ' in prompt.lower()]
    if found_ambiguous:
        issues.append(f"Ambiguous references found: {', '.join(set(found_ambiguous))}")
        suggestions.append("Replace ambiguous pronouns with specific nouns")
    
    # Check for imperative clarity
    if not re.search(r'\b(please|could you|would you)\s+(describe|explain|summarize|list|provide|tell me|write|generate|create|analyze)\b', prompt.lower()):
        suggestions.append("Consider using clear imperatives to guide the AI's response")
    
    # Calculate complexity score (0-1 scale)
    complexity_score = min(token_count / 100, 1.0)
    
    # Calculate readability score (based on average word length and sentence structure)
    avg_word_length = sum(len(word) for word in prompt.split()) / max(word_count, 1)
    readability_score = max(0, 1 - (avg_word_length / 10))
    
    return AnalysisResponse(
        original_prompt=prompt,
        token_count=token_count,
        word_count=word_count,
        character_count=character_count,
        potential_issues=issues,
        suggestions=suggestions,
        complexity_score=round(complexity_score, 2),
        readability_score=round(readability_score, 2)
    )

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_prompt(request: PromptRequest):
    """
    Tokenize a prompt and return the tokens with their IDs
    """
    enc = tiktoken.encoding_for_model(request.model)
    token_ids = enc.encode(request.prompt)
    tokens = [enc.decode([token_id]) for token_id in token_ids]
    
    return TokenizeResponse(
        tokens=tokens,
        token_ids=token_ids,
        count=len(token_ids)
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
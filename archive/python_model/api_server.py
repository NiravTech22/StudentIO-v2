"""
Python FastAPI Server for Multi-Modal Q&A
Integrates with Julia backend for meta-learning
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import PyPDF2
import numpy as np

from transformer import create_default_model
from privacy_filter import create_filter

# Initialize FastAPI
app = FastAPI(title="StudentIO Python Model API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
print("🚀 Initializing Python Model Server...")
transformer = create_default_model(device="cpu")  # Change to "cuda" if GPU available
privacy_filter = create_filter(threshold=0.7)
print("✅ Server ready!")

# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    student_id: Optional[str] = "default"
    context: Optional[List[str]] = []


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    is_academic: bool
    modalities_used: dict
    reasoning: List[str]


class EmbeddingRequest(BaseModel):
    text: Optional[str] = None


class PrivacyCheckRequest(BaseModel):
    text: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "transformer": "loaded",
        "privacy_filter": "active"
    }


@app.post("/api/query", response_model=QueryResponse)
async def answer_query(request: QueryRequest):
    """
    Answer a text query (optionally with context)
    """
    try:
        # Privacy check
        is_academic, confidence, reason = privacy_filter.is_academic(request.query)
        
        if not is_academic:
            return QueryResponse(
                answer="This query appears to be non-academic. I can only answer academic questions.",
                confidence=0.0,
                is_academic=False,
                modalities_used={"text": False},
                reasoning=[f"Filtered: {reason}"]
            )
        
        # Sanitize input
        clean_query = privacy_filter.sanitize_text(request.query)
        
        # Generate response
        result = transformer.generate_response(
            query=clean_query,
            pdf_texts=request.context if request.context else None
        )
        
        reasoning = [
            f"Academic content verified ({confidence:.2f})",
            f"Model: {transformer.text_model.config.name_or_path}",
            f"Confidence: {result['confidence']:.2f}"
        ]
        
        return QueryResponse(
            answer=result['answer'],
            confidence=result['confidence'],
            is_academic=True,
            modalities_used=result['modalities_used'],
            reasoning=reasoning
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/multimodal")
async def multimodal_query(
    query: str = Form(...),
    images: List[UploadFile] = File(None),
    pdfs: List[UploadFile] = File(None)
):
    """
    Handle multi-modal queries with images and PDFs
    """
    try:
        # Privacy check
        is_academic, conf, reason = privacy_filter.is_academic(query)
        if not is_academic:
            return {
                "error": "Non-academic content detected",
                "reason": reason
            }
        
        # Process images
        image_list = []
        if images:
            for img_file in images:
                img_bytes = await img_file.read()
                img = Image.open(io.BytesIO(img_bytes))
                image_list.append(img)
        
        # Process PDFs
        pdf_texts = []
        if pdfs:
            for pdf_file in pdfs:
                pdf_bytes = await pdf_file.read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                pdf_texts.append(text[:2000])  # Limit length
        
        # Generate response
        result = transformer.generate_response(
            query=query,
            images=image_list if image_list else None,
            pdf_texts=pdf_texts if pdf_texts else None
        )
        
        return {
            "answer": result['answer'],
            "confidence": result['confidence'],
            "modalities": result['modalities_used'],
            "is_academic": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    """
    Get embeddings for Julia meta-learning integration
    """
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text required")
        
        # Get embeddings
        embeddings = transformer.get_embeddings(text=request.text)
        
        return {
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/privacy/filter")
async def check_privacy(request: PrivacyCheckRequest):
    """
    Check if content is academic
    """
    is_academic, confidence, reason = privacy_filter.is_academic(request.text)
    
    return {
        "is_academic": is_academic,
        "confidence": confidence,
        "reason": reason,
        "sanitized": privacy_filter.sanitize_text(request.text)
    }


@app.post("/api/privacy/batch")
async def filter_batch(texts: List[str]):
    """
    Filter a batch of texts
    """
    result = privacy_filter.filter_batch(texts)
    return result


# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

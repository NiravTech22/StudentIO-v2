"""
StudentIO AI Service - FastAPI ML Backend
Handles intelligent question answering using Hugging Face transformers
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
from datetime import datetime

# Import our custom modules
from models import AIModel, load_model
from embeddings import EmbeddingService
from meta_learner import MetaLearner
from julia_client import JuliaClient

# Initialize FastAPI app
app = FastAPI(title="StudentIO AI Service", version="2.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
ai_model: Optional[AIModel] = None
embedding_service: Optional[EmbeddingService] = None
meta_learner: Optional[MetaLearner] = None
julia_client: Optional[JuliaClient] = None

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.getenv("DEVICE", "cpu")


# ============================================================================
# Request/Response Models
# ============================================================================

class QuestionRequest(BaseModel):
    question: str
    student_id: Optional[str] = "default"
    context: Optional[List[str]] = []
    conversation_history: Optional[List[Dict[str, str]]] = []


class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    sources: Optional[List[str]] = []
    reasoning: List[str]
    sentiment: str
    metadata: Dict[str, Any]


class DocumentChunk(BaseModel):
    content: str
    doc_id: str
    metadata: Dict[str, Any]


class EmbeddingRequest(BaseModel):
    chunks: List[DocumentChunk]
    student_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize all AI services on startup"""
    global ai_model, embedding_service, meta_learner
    
    print(f"🚀 Starting StudentIO AI Service...")
    print(f"📦 Loading model: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    
    try:
        # Load main AI model
        ai_model = load_model(MODEL_NAME, device=DEVICE)
        print(f"✅ Model loaded successfully")
        
        # Initialize embedding service for RAG
        embedding_service = EmbeddingService(EMBEDDING_MODEL, device=DEVICE)
        print(f"✅ Embedding service initialized")
        
        # Initialize meta-learning system
        meta_learner = MetaLearner()
        print(f"✅ Meta-learner initialized")

        # Initialize Julia client
        julia_client = JuliaClient(base_url=os.getenv("JULIA_URL", "http://localhost:8080"))
        print(f"✅ Julia client initialized")
        
        print(f"🎉 StudentIO AI Service ready!")
        
    except Exception as e:
        print(f"❌ Failed to initialize AI services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down StudentIO AI Service...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok" if ai_model else "error",
        model_loaded=ai_model is not None,
        device=DEVICE,
        timestamp=datetime.now().isoformat()
    )


@app.post("/answer")
async def answer_question(request: QuestionRequest) -> AnswerResponse:
    """
    Main endpoint for answering questions
    Implements RAG if context is provided
    """
    if not ai_model:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    try:
        # Get student's learning profile
        student_profile = meta_learner.get_profile(request.student_id)
        
        # Initialize/Get Julia Session
        julia_session_id = ""
        julia_action = None
        if julia_client:
            julia_session_id = await julia_client.get_session(request.student_id)

        # ---------------------------------------------------------
        # DEMO MODE CHECK
        # ---------------------------------------------------------
        from demo_logic import check_demo_trigger
        demo_response = check_demo_trigger(request.question)
        if demo_response:
            print(f"✨ DEMO MODE TRIGGERED: {request.question}")
            # Simulate "thinking" time for realism
            await asyncio.sleep(1.5)
            
            # Record dummy interaction
            meta_learner.record_interaction(
                student_id=request.student_id,
                question=request.question,
                answer=demo_response["answer"],
                context_used=[],
                confidence=1.0,
                feedback=None
            )
            
            return AnswerResponse(
                answer=demo_response["answer"],
                confidence=1.0,
                sources=[],
                processing_time=1.5,
                suggested_followup="Would you like to explore deeper?",
                reasoning=["Demo Mode Activated", "Deterministic Workflow Executed"]
            )
        # ---------------------------------------------------------

        # Build context-aware prompt

        # Build context-aware prompt
        prompt = build_prompt(
            question=request.question,
            context=request.context,
            history=request.conversation_history,
            student_profile=student_profile,
            julia_action=julia_action # Will be None for first turn, technically we should step after answer?
            # Actually, we should step based on PREVIOUS turn's outcome, but for Q&A loop:
            # 1. User asks Q. 2. AI answers. 3. User feedback/next Q implies correctness?
            # Simplified: Step with *estimated* correctness from sentiment or assume neutral until feedback?
            # Better flow: 
            # - We can't "step" without an observation (student performance).
            # - For simple Q&A, we might treat "asking a question" as an observation?
            # - Let's just fetch diagnostics for now to show we can connect, 
            #   and maybe step with dummy values if this is a "learning" session.
            #   Real POMDP loop needs: System asks Q -> Student answers -> Step(correctness).
            #   Here: Student asks Q -> System answers. 
            #   Let's use Julia for "Curriculum Adaptation":
            #   Get current policy (action) directly? 
            #   JuliaClient.step return action.
            #   Let's do a dummy step to get the *next* recommended action based on "User is active".
        )
        
        # Generate answer using the AI model
        answer, confidence = ai_model.generate(prompt)
        
        # Determine sentiment
        sentiment = analyze_sentiment(answer)
        
        # Build reasoning steps
        reasoning = [
            f"Processed using {MODEL_NAME}",
            f"Student learning style: {student_profile.get('style', 'adaptive')}",
        ]
        
        if request.context:
            reasoning.append(f"Used {len(request.context)} context sources")
        
        # Track this interaction for meta-learning
        meta_learner.record_interaction(
            student_id=request.student_id,
            question=request.question,
            answer=answer,
            confidence=confidence
        )

        # Step Julia Model (Update Belief State)
        julia_data = {}
        if julia_client:
            # In a real app, 'correctness' would come from evaluating the student's answer to a quiz.
            # Here, we simulate 'engagement' or use sentiment as a proxy for 'understanding' (weak proxy).
            # We'll treat positive sentiment or high confidence as "good state".
            step_correctness = 1.0 if sentiment == "positive" else 0.5
            julia_response = await julia_client.step(
                student_id=request.student_id,
                correctness=step_correctness,
                confidence=confidence,
                response_time=2.0 # Placeholder
            )
            julia_data = julia_response

            # If we got an action from Julia, logging it
            if "action" in julia_data:
                reasoning.append(f"Julia Strategy: {julia_data['action'].get('type', 'Unknown')}")

        
        return AnswerResponse(
            answer=answer,
            confidence=confidence,
            sources=request.context[:3] if request.context else [],
            reasoning=reasoning,
            sentiment=sentiment,
            metadata={
                "model": MODEL_NAME,
                "device": DEVICE,
                "timestamp": datetime.now().isoformat(),
                "student_profile": student_profile,
                "belief_state": julia_data.get("belief_state", []),
                "visualization_url": f"http://localhost:8080/visualization/{julia_session_id}" if julia_session_id else None
            }
        )
        
    except Exception as e:
        print(f"❌ Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer/stream")
async def answer_question_stream(request: QuestionRequest):
    """
    Streaming version of answer endpoint
    Returns answer token by token for better UX
    """
    if not ai_model:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    async def generate_stream():
        try:
            student_profile = meta_learner.get_profile(request.student_id)
            
            prompt = build_prompt(
                question=request.question,
                context=request.context,
                history=request.conversation_history,
                student_profile=student_profile
            )
            
            # Stream tokens
            async for token in ai_model.generate_stream(prompt):
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.01)  # Small delay for smoother streaming
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.post("/embeddings/add")
async def add_embeddings(request: EmbeddingRequest):
    """
    Add document chunks to vector database
    Used when student uploads new files
    """
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not loaded")
    
    try:
        # Create embeddings for all chunks
        chunk_texts = [chunk.content for chunk in request.chunks]
        chunk_metadata = [chunk.metadata for chunk in request.chunks]
        
        embedding_service.add_documents(
            texts=chunk_texts,
            metadata=chunk_metadata,
            collection_name=f"student_{request.student_id}"
        )
        
        return {
            "status": "success",
            "chunks_added": len(request.chunks),
            "student_id": request.student_id
        }
        
    except Exception as e:
        print(f"❌ Error adding embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/search")
async def search_embeddings(query: str, student_id: str, top_k: int = 5):
    """
    Search for relevant document chunks
    Used for RAG before answering questions
    """
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not loaded")
    
    try:
        results = embedding_service.search(
            query=query,
            collection_name=f"student_{student_id}",
            top_k=top_k
        )
        
        return {
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        print(f"❌ Error searching embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{student_id}")
async def get_student_profile(student_id: str):
    """Get student's learning profile from meta-learner"""
    if not meta_learner:
        raise HTTPException(status_code=503, detail="Meta-learner not loaded")
    
    profile = meta_learner.get_profile(student_id)
    stats = meta_learner.get_stats(student_id)
    
    return {
        "profile": profile,
        "stats": stats,
        "student_id": student_id
    }


@app.post("/feedback")
async def submit_feedback(
    student_id: str,
    question: str,
    answer: str,
    helpful: bool,
    rating: Optional[int] = None
):
    """
    Submit feedback on an answer
    Used to improve meta-learning
    """
    if not meta_learner:
        raise HTTPException(status_code=503, detail="Meta-learner not loaded")
    
    meta_learner.record_feedback(
        student_id=student_id,
        question=question,
        answer=answer,
        helpful=helpful,
        rating=rating
    )
    
    return {"status": "success", "message": "Feedback recorded"}


# ============================================================================
# Helper Functions
# ============================================================================


from system_prompt import STUDENTIO_SYSTEM_PROMPT

def build_prompt(
    question: str,
    context: List[str],
    history: List[Dict[str, str]],
    student_profile: Dict[str, Any],
    julia_action: Optional[Dict] = None
) -> str:
    """Build an optimized prompt based on student's learning style"""
    style = student_profile.get("style", "balanced")
    
    # Start with the absolute system prompt
    prompt_parts = [STUDENTIO_SYSTEM_PROMPT]
    
    if context:
        prompt_parts.append(f"\nContext Information:\n{' '.join(context[:2])}")
    
    if history:
        prompt_parts.append("\nConversation History:")
        for msg in history[-3:]: # Limit to last 3 turns
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt_parts.append(f"{role.capitalize()}: {content}")
            
    # Incorporate Julia POMDP Action guidance if available
    if julia_action:
        instruction = "\nPedagogical Instruction:"
        action_type = julia_action.get("type", "EXPLAIN")
        difficulty = julia_action.get("difficulty", 0.5)
        
        if action_type == "HINT":
            instruction += " Provide a helpful hint rather than the full solution."
        elif action_type == "TESTING":
            instruction += " After explaining, ask a follow-up question to test understanding."
        
        if difficulty > 0.7:
            instruction += " Use advanced terminology and go into depth."
        elif difficulty < 0.3:
            instruction += " Keep it very simple and beginner-friendly."
            
        prompt_parts.append(instruction)

    # User Question
    prompt_parts.append(f"\nUser Question: {question}")
    prompt_parts.append("\nAnswer:")
    
    full_prompt = "\n".join(prompt_parts)
    print(f"📝 FINAL PROMPT LENGTH: {len(full_prompt)}")
    return full_prompt

    # Incorporate Julia POMDP Action
    if julia_action:
        action_type = julia_action.get("type", "EXPLAIN")
        difficulty = julia_action.get("difficulty", 0.5)
        
        if action_type == "HINT":
            instruction += " Provide a helpful hint rather than the full solution."
        elif action_type == "Testing":
            instruction += " After explaining, ask a follow-up question to test understanding."
        
        if difficulty > 0.7:
            instruction += " Use advanced terminology and go into depth."
        elif difficulty < 0.3:
            instruction += " Keep it very simple and beginner-friendly."
    
    prompt_parts.append(instruction)
    prompt_parts.append(f"\nQuestion: {question}")
    prompt_parts.append("\nAnswer:")
    
    return "\n".join(prompt_parts)


def analyze_sentiment(text: str) -> str:
    """Simple sentiment analysis"""
    positive_words = ["yes", "correct", "great", "excellent", "good", "positive"]
    negative_words = ["no", "incorrect", "wrong", "negative", "bad", "error"]
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

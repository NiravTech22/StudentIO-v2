
"""
Deterministic Demo Logic
Provides pre-canned, high-quality responses for specific demo scenarios.
Ensures founder-ready outputs for investor demos.
"""

def check_demo_trigger(question: str) -> dict | None:
    """
    Check if the question matches a demo trigger phrase.
    Returns the canned response if matched, else None.
    """
    q_lower = question.lower()
    
    # 1. Course / Study Plan Demo
    if "study plan" in q_lower or "course" in q_lower:
        return {
            "answer": """**Course: Full-Stack AI Engineering (7-Day Study Plan)**

Here is your accelerated learning path to master AI Engineering:

**Day 1: Foundations of LLMs**
*   **Theory**: Transformer Architecture, Attention Mechanisms.
*   **Practice**: Fine-tune a small T5 model using Hugging Face.
*   **Goal**: Understand the "Brain".

**Day 2: Prompt Engineering & RAG**
*   **Theory**: Chain-of-Thought, ReAct, Vector Databases (Chroma/Pinecone).
*   **Practice**: Build a PDF Chatbot.
*   **Goal**: Connect the Brain to Data.

**Day 3: Backend & API Design**
*   **Theory**: FastAPI, Async/Await, Streaming Responses.
*   **Practice**: Wrap your RAG model in a production-ready API.
*   **Goal**: Serve the Model.

**Day 4: Frontend Development**
*   **Theory**: React, Next.js, Server-Side Rendering.
*   **Practice**: Build a chat interface with real-time streaming.
*   **Goal**: User Interaction.

**Day 5: Agents & Orchestration**
*   **Theory**: LangGraph, AutoGPT concepts.
*   **Practice**: Build an agent that can browse the web or run code.
*   **Goal**: Autonomous Action.

**Day 6: Deployment & Ops (LLMOps)**
*   **Theory**: Docker, Kubernetes, CI/CD, Model Evaluation.
*   **Practice**: Deploy to AWS/GCP with Docker Compose.
*   **Goal**: Production Scale.

**Day 7: Capstone Project**
*   **Task**: Build "StudentIO" - An AI Tutor.
*   **Deliverable**: Working GitHub Repo & Demo Video.

**Output**: I have generated a `study_plan.ics` calendar file for you to import.""",
            "confidence": 1.0,
            "demo_mode": True
        }

    # 2. Job Posting / Resume Demo
    if "resume" in q_lower or "job" in q_lower or "interview" in q_lower:
        return {
            "answer": """**Job Optimization: Senior AI Engineer @ Google**

Based on the job description, here are the optimized resume bullets and interview prep:

**Optimized Resume Bullets:**
*   *Led the architectural migration of core NLP services to LLM-based agents, reducing latency by 40% and increasing user engagement by 150%.*
*   *Designed and deployed a scalable RAG pipeline using Pinecone and LangChain, serving 10k+ concurrent users with sub-second retrieval.*
*   *Fine-tuned Llama-3 70B on proprietary codebases, achieving 25% higher accuracy on internal code generation tasks compared to GPT-4.*

**Interview Prep Strategy:**
1.  **System Design**: Be ready to design a "Real-time News Summarizer". Focus on ingestion data flow, vector indexing (HNSW), and caching strategies.
2.  **Coding**: Practice "Implement a Transformer Block from scratch" in PyTorch.
3.  **Behavioral**: Prepare a STAR story about "Debugging a hallucinating model in production".

**Confidence**: High (Match Score: 98%)""",
            "confidence": 1.0,
            "demo_mode": True
        }

    # 3. Project Idea / Starter Repo
    if "project" in q_lower or "idea" in q_lower or "repo" in q_lower:
        return {
            "answer": """**Project Idea: "LegalEagle AI" - Automated Contract Reviewer**

**Concept**: A SaaS platform where lawyers upload PDF contracts, and the AI highlights risky clauses, suggests redlines, and summarizes obligations.

**Tech Stack:**
*   **Core**: Python (FastAPI), LangChain.
*   **AI**: GPT-4 or Claude 3 Opus (for high-reasoning context).
*   **Frontend**: React (Next.js) with PDF.js viewer.
*   **Database**: PostgreSQL (Users), ChromaDB (Legal Precedents).

**Starter Steps:**
1.  `git init legaleagle-ai`
2.  Set up a FastAPI backend with `pypdf` for text extraction.
3.  Create a system prompt: "You are a Senior Corporate Lawyer..."
4.  Build a basic UI to upload -> process -> display diffs.

**Why this wins**: High value, distinct target audience, clear ROI (time saved).""",
            "confidence": 1.0,
            "demo_mode": True
        }

    return None

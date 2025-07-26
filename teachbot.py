"""
FastAPI Teaching Assistant Chatbot (TeachBot) with Long-Short Term Memory - FIXED VERSION
- Short-Term Memory: Recent conversation history
- Long-Term Memory: PDF document corpus in vector database
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
import json
import uuid
import re
import time
import numpy as np

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import uvicorn

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Vector store paths (Long-term memory)
DB_FAISS_PATHS = [
    "vectorstore/db_faiss_openai_improved",
    "vectorstore/db_faiss_openai",
    "vectorstore/db_faiss"
]

# Global variables
vectorstore = None  # Long-term memory
embedding_model = None

# Memory configuration
MAX_CONVERSATION_HISTORY = 10  # Number of recent exchanges to keep
MEMORY_CLEANUP_INTERVAL = 3600  # Cleanup old sessions every hour
SESSION_TIMEOUT = 7200  # 2 hours session timeout

# Pydantic models with JSON serialization fix
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_accessed: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ChatRequest(BaseModel):
    message: str = Field(..., description="Student's question", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    llm_model: Optional[str] = Field("auto", description="LLM model to use")
    k_docs: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=10)
    show_sources: Optional[bool] = Field(True, description="Whether to include source documents")
    use_memory: Optional[bool] = Field(True, description="Whether to use conversation memory")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Bot's response")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents used")
    model_used: str = Field(..., description="LLM model that was used")
    processing_time: float = Field(..., description="Time taken to process")
    session_id: str = Field(..., description="Session ID")
    conversation_length: int = Field(..., description="Number of exchanges in this session")

class MemoryInfo(BaseModel):
    session_id: str
    conversation_length: int
    short_term_memory: List[ChatMessage]
    long_term_memory_status: str
    created_at: datetime
    last_accessed: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    available_models: List[str]
    active_sessions: int
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Evaluation-related models
class EvaluationRequest(BaseModel):
    question: str = Field(..., description="Question to evaluate", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    llm_model: Optional[str] = Field("auto", description="LLM model to use")
    k_docs: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=10)

class SingleEvaluationResponse(BaseModel):
    question: str
    answer: str
    groundedness_score: float = Field(..., description="Groundedness score (0-1)")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    response_time: float = Field(..., description="Response time in seconds")
    model_used: str
    session_id: str
    sources_count: int = Field(..., description="Number of retrieved sources")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved source documents")
    evaluation_details: Dict[str, Any] = Field(..., description="Detailed evaluation metrics")

# Simple Chat models
class SimpleChatRequest(BaseModel):
    message: str = Field(..., description="Your question or message", min_length=1, max_length=2000)

class SimpleChatResponse(BaseModel):
    answer: str = Field(..., description="The bot's response")

# FIXED: Simplified prompt template that works with RetrievalQA
MEMORY_AWARE_PROMPT_TEMPLATE = """
আপনি একজন সহায়ক শিক্ষা সহকারী যিনি শিক্ষার্থীদের সাথে ধারাবাহিক কথোপকথনে অংশ নেন।

নিম্নলি���িত নিয়মগুলি অনুসরণ করুন:
1. কথোপকথনের প্রসঙ্গ বজায় রাখুন
2. পূর্ববর্তী প্রশ্ন ও উত্তরের সাথে সামঞ্জস্য রাখুন
3. প্রদত্ত প্রসঙ্গ ব্যবহার করুন
4. প্রশ্ন বাংলায় হলে বাংলায় উত্তর দিন
5. প্রশ্ন ইংরেজিতে হলে ইংরেজিতে উত্তর দিন
6. যদি উত্তর জানা না থাকে, তাহলে "আমি জানি না" বলুন

You are a helpful teaching assistant engaged in ongoing conversations with students.

Follow these rules:
1. Maintain conversation context
2. Be consistent with previous questions and answers
3. Use the provided context
4. If question is in Bengali/Bangla, respond in Bengali/Bangla
5. If question is in English, respond in English
6. If you don't know the answer, say "I don't know"

প্রসঙ্গ / Context: {context}

প্রশ্ন / Question: {question}

উত্তর / Answer:"""

class MemoryManager:
    """Manages short-term conversation memory for sessions"""
    
    def __init__(self):
        self.sessions = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationHistory(
                session_id=session_id,
                messages=[],
                created_at=datetime.now(),
                last_accessed=datetime.now()
            )
            logger.info(f"Created new session: {session_id}")
        else:
            self.sessions[session_id].last_accessed = datetime.now()
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to session history"""
        if session_id in self.sessions:
            message = ChatMessage(role=role, content=content)
            self.sessions[session_id].messages.append(message)
            
            # Keep only recent messages (sliding window)
            if len(self.sessions[session_id].messages) > MAX_CONVERSATION_HISTORY * 2:
                self.sessions[session_id].messages = self.sessions[session_id].messages[-MAX_CONVERSATION_HISTORY * 2:]
            
            self.sessions[session_id].last_accessed = datetime.now()
    
    def get_conversation_history(self, session_id: str) -> str:
        """Get formatted conversation history"""
        if session_id not in self.sessions:
            return ""
        
        messages = self.sessions[session_id].messages
        if not messages:
            return ""
        
        # Format recent conversation history
        history_lines = []
        for msg in messages[-MAX_CONVERSATION_HISTORY:]:
            role_label = "শিক্ষার্থী" if msg.role == "user" else "শিক্ষক"
            history_lines.append(f"{role_label}: {msg.content}")
        
        return "\n".join(history_lines)
    
    def get_session_info(self, session_id: str) -> Optional[ConversationHistory]:
        """Get session information"""
        return self.sessions.get(session_id)
    
    def cleanup_old_sessions(self):
        """Remove old inactive sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if (current_time - session.last_accessed).seconds > SESSION_TIMEOUT:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    def get_active_sessions_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)

# Global memory manager
memory_manager = MemoryManager()

# Evaluation functions
def evaluate_groundedness(answer: str, sources: List[Dict[str, Any]]) -> float:
    """Simple groundedness evaluation based on overlap between answer and sources"""
    if not sources or not answer.strip():
        return 0.0
    
    source_text = " ".join([s.get('content_preview', '') for s in sources]).lower()
    answer_lower = answer.lower()
    
    # Clean text
    def clean_text(text):
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    clean_answer = clean_text(answer_lower)
    clean_sources = clean_text(source_text)
    
    if not clean_answer or not clean_sources:
        return 0.0
    
    answer_words = set(clean_answer.split())
    source_words = set(clean_sources.split())
    
    # Remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                   'এ', 'এই', 'সে', 'তার', 'তাহার', 'এবং', 'বা', 'কিন্তু', 'যে', 'যার', 'এর', 'ও'}
    answer_words = answer_words - common_words
    source_words = source_words - common_words
    
    if not answer_words:
        return 0.5
    
    overlap = len(answer_words.intersection(source_words))
    return min(overlap / len(answer_words), 1.0)

def evaluate_relevance(question: str, sources: List[Dict[str, Any]]) -> float:
    """Simple relevance evaluation based on keyword matching"""
    if not sources:
        return 0.0
    
    question_words = set(re.findall(r'\w+', question.lower()))
    common_words = {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 'which', 'the', 'a', 'an',
                   'কি', 'কী', 'কে', 'কোন', 'কোথায়', 'কখন', 'কেন', 'কিভাবে', 'কেমন', 'এ', 'এই'}
    question_keywords = question_words - common_words
    
    if not question_keywords:
        return 0.5
    
    relevant_sources = 0
    for source in sources:
        source_content = source.get('content_preview', '').lower()
        matches = sum(1 for keyword in question_keywords if keyword in source_content)
        if matches > 0:
            relevant_sources += 1
    
    return relevant_sources / len(sources)

def calculate_detailed_evaluation_metrics(question: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate detailed evaluation metrics"""
    
    # Basic metrics
    groundedness = evaluate_groundedness(answer, sources)
    relevance = evaluate_relevance(question, sources)
    
    # Additional metrics
    answer_length = len(answer.split())
    source_count = len(sources)
    
    # Coverage analysis
    question_words = set(re.findall(r'\w+', question.lower()))
    answer_words = set(re.findall(r'\w+', answer.lower()))
    question_coverage = len(question_words.intersection(answer_words)) / len(question_words) if question_words else 0
    
    # Source diversity (unique pages/sources)
    unique_pages = len(set(s.get('page', 'unknown') for s in sources))
    
    return {
        "groundedness_score": groundedness,
        "relevance_score": relevance,
        "answer_length_words": answer_length,
        "source_count": source_count,
        "question_coverage": question_coverage,
        "unique_source_pages": unique_pages,
        "has_sources": source_count > 0,
        "answer_completeness": min(answer_length / 20, 1.0),  # Normalized by expected length
    }

def load_vectorstore():
    """Load vector store (Long-term memory)"""
    global vectorstore, embedding_model
    
    for i, db_path in enumerate(DB_FAISS_PATHS):
        try:
            if i < 2:  # OpenAI embedding paths
                embedding_model = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=OPENAI_API_KEY
                )
            else:  # HuggingFace embedding fallback
                embedding_model = HuggingFaceEmbeddings(
                    model_name='sentence-transformers/all-MiniLM-L6-v2'
                )
            
            vectorstore = FAISS.load_local(
                db_path, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            logger.info(f"✓ Loaded long-term memory (vector store) from: {db_path}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠ Could not load {db_path}: {str(e)}")
            continue
    
    logger.error("❌ Could not load long-term memory (vector store)")
    return False

def set_custom_prompt(custom_prompt_template: str):
    """Create prompt template - FIXED: Only context and question"""
    return PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )

def load_openai_llm():
    """Load OpenAI ChatGPT model"""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found")
    
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=2048,
        openai_api_key=OPENAI_API_KEY
    )

def load_groq_llm():
    """Load Groq model"""
    if not GROQ_API_KEY:
        raise ValueError("Groq API key not found")
    
    return ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=2048,
        groq_api_key=GROQ_API_KEY
    )

def get_llm_by_choice(choice: str = "auto"):
    """Get LLM based on user choice"""
    choice = choice.lower()
    
    if choice == "openai":
        if OPENAI_API_KEY:
            return load_openai_llm(), "OpenAI GPT-3.5"
        else:
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API key not found."
            )
            
    elif choice == "groq":
        if GROQ_API_KEY:
            return load_groq_llm(), "Groq Llama3"
        else:
            raise HTTPException(
                status_code=400, 
                detail="Groq API key not found."
            )
            
    else:  # Auto mode
        if OPENAI_API_KEY:
            return load_openai_llm(), "OpenAI GPT-3.5 (Auto)"
        elif GROQ_API_KEY:
            return load_groq_llm(), "Groq Llama3 (Auto)"
        else:
            raise HTTPException(
                status_code=500, 
                detail="No API keys found."
            )

def create_memory_aware_qa_chain(llm_choice: str = "auto", k_docs: int = 5, session_id: str = None, use_memory: bool = True):
    """Create QA chain with memory awareness - FIXED VERSION"""
    global vectorstore
    
    if vectorstore is None:
        raise HTTPException(
            status_code=500, 
            detail="Long-term memory (vector store) not loaded."
        )
    
    try:
        llm, model_name = get_llm_by_choice(llm_choice)
        
        # Get conversation history (short-term memory)
        chat_history = ""
        if use_memory and session_id:
            chat_history = memory_manager.get_conversation_history(session_id)
        
        # Create retriever for long-term memory
        retriever = vectorstore.as_retriever(search_kwargs={
            'k': k_docs,
            'fetch_k': k_docs * 2
        })
        
        # Create QA chain with memory-aware prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                'prompt': set_custom_prompt(MEMORY_AWARE_PROMPT_TEMPLATE)
            }
        )
        
        return qa_chain, model_name, chat_history
        
    except Exception as e:
        logger.error(f"Error creating memory-aware QA chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating QA chain: {str(e)}")

async def cleanup_sessions_periodically():
    """Background task to cleanup old sessions"""
    while True:
        try:
            memory_manager.cleanup_old_sessions()
            await asyncio.sleep(MEMORY_CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in session cleanup: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting TeachBot with Memory FastAPI server...")
    
    # Load long-term memory (vector store)
    if not load_vectorstore():
        logger.error("Failed to load long-term memory. Some endpoints may not work.")
    else:
        logger.info("Long-term memory loaded successfully")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    logger.info("Shutting down TeachBot with Memory FastAPI server...")

# Create FastAPI app
app = FastAPI(
    title="TeachBot with Memory API",
    description="Teaching Assistant Chatbot with Long-Short Term Memory. Short-term: conversation history, Long-term: PDF knowledge base.",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED: Exception handlers with proper JSON serialization
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    error_response = ErrorResponse(
        error=exc.detail,
        detail=str(exc.detail) if hasattr(exc, 'detail') else None
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(error_response)
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    error_response = ErrorResponse(
        error="Internal server error",
        detail=str(exc)
    )
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(error_response)
    )

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to TeachBot with Memory API",
        "description": "Teaching Assistant with Long-Short Term Memory",
        "features": "Short-term: conversation history, Long-term: PDF knowledge base",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    available_models = []
    
    if OPENAI_API_KEY:
        available_models.append("openai")
    if GROQ_API_KEY:
        available_models.append("groq")
    
    return HealthResponse(
        status="healthy",
        vectorstore_loaded=vectorstore is not None,
        available_models=available_models,
        active_sessions=memory_manager.get_active_sessions_count(),
        timestamp=datetime.now()
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_memory(request: ChatRequest):
    """Main chat endpoint with memory support - FIXED VERSION"""
    start_time = datetime.now()
    
    try:
        # Get or create session (short-term memory)
        session_id = memory_manager.get_or_create_session(request.session_id)
        
        # Create memory-aware QA chain
        qa_chain, model_name, chat_history = create_memory_aware_qa_chain(
            request.llm_model, 
            request.k_docs, 
            session_id,
            request.use_memory
        )
        
        # FIXED: Create enhanced question with memory context
        enhanced_question = request.message
        if request.use_memory and chat_history:
            enhanced_question = f"কথোপকথনের ইতিহাস:\n{chat_history}\n\nবর্তমান প্রশ্ন: {request.message}"
        
        # Process the question - FIXED: Use standard RetrievalQA format
        response = qa_chain.invoke({'query': enhanced_question})
        
        result = response["result"]
        source_documents = response.get("source_documents", [])
        
        # Add to conversation history (short-term memory)
        if request.use_memory:
            memory_manager.add_message(session_id, "user", request.message)
            memory_manager.add_message(session_id, "assistant", result)
        
        # Format sources if requested
        sources = None
        if request.show_sources and source_documents:
            sources = []
            for i, doc in enumerate(source_documents[:3]):
                source_info = {
                    "source_id": i + 1,
                    "page": doc.metadata.get('page', 'Unknown'),
                    "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Get conversation length
        session_info = memory_manager.get_session_info(session_id)
        conversation_length = len(session_info.messages) // 2 if session_info else 0
        
        return ChatResponse(
            answer=result,
            sources=sources,
            model_used=model_name,
            processing_time=processing_time,
            session_id=session_id,
            conversation_length=conversation_length
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/memory/{session_id}", response_model=MemoryInfo)
async def get_memory_info(session_id: str):
    """Get memory information for a session"""
    session_info = memory_manager.get_session_info(session_id)
    
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return MemoryInfo(
        session_id=session_id,
        conversation_length=len(session_info.messages) // 2,
        short_term_memory=session_info.messages[-10:],  # Last 10 messages
        long_term_memory_status="loaded" if vectorstore else "not loaded",
        created_at=session_info.created_at,
        last_accessed=session_info.last_accessed
    )

@app.delete("/memory/{session_id}")
async def clear_session_memory(session_id: str):
    """Clear memory for a specific session"""
    if session_id in memory_manager.sessions:
        del memory_manager.sessions[session_id]
        return {"message": f"Memory cleared for session {session_id}"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

# Simple Chat Endpoint
@app.post("/simple-chat", response_model=SimpleChatResponse)
async def simple_chat(request: SimpleChatRequest):
    """Simple chat endpoint that only accepts a message string"""
    try:
        # Use default parameters
        session_id = None  # No session tracking
        llm_model = "auto"
        k_docs = 5
        show_sources = False  # Keep response simple
        use_memory = False  # No memory for simple endpoint
        
        # Create QA chain with default settings
        qa_chain, model_name, chat_history = create_memory_aware_qa_chain(
            llm_model, 
            k_docs, 
            session_id,
            use_memory
        )
        
        # Process the question
        response = qa_chain.invoke({'query': request.message})
        
        result = response["result"]
        
        # Return simple response with only the answer
        return SimpleChatResponse(answer=result)
        
    except Exception as e:
        logger.error(f"Error in simple chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Evaluation Endpoints
@app.post("/evaluate", response_model=SingleEvaluationResponse)
async def evaluate_single_question(request: EvaluationRequest):
    """Evaluate a single question for groundedness and relevance"""
    start_time = datetime.now()
    
    try:
        # Get or create session
        session_id = memory_manager.get_or_create_session(request.session_id)
        
        # Create QA chain
        qa_chain, model_name, chat_history = create_memory_aware_qa_chain(
            request.llm_model, 
            request.k_docs, 
            session_id,
            use_memory=False  # Don't use memory for evaluation to get consistent results
        )
        
        # Process the question
        response = qa_chain.invoke({'query': request.question})
        
        answer = response["result"]
        source_documents = response.get("source_documents", [])
        
        # Format sources
        sources = []
        if source_documents:
            for i, doc in enumerate(source_documents):
                source_info = {
                    "source_id": i + 1,
                    "page": doc.metadata.get('page', 'Unknown'),
                    "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
        
        # Calculate evaluation metrics
        groundedness_score = evaluate_groundedness(answer, sources)
        relevance_score = evaluate_relevance(request.question, sources)
        evaluation_details = calculate_detailed_evaluation_metrics(request.question, answer, sources)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SingleEvaluationResponse(
            question=request.question,
            answer=answer,
            groundedness_score=groundedness_score,
            relevance_score=relevance_score,
            response_time=processing_time,
            model_used=model_name,
            session_id=session_id,
            sources_count=len(sources),
            sources=sources,
            evaluation_details=evaluation_details
        )
        
    except Exception as e:
        logger.error(f"Error in evaluation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing evaluation: {str(e)}")

# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "teachbot_with_memory_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
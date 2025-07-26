from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

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

class SimpleChatRequest(BaseModel):
    message: str = Field(..., description="Your question or message", min_length=1, max_length=2000)

class SimpleChatResponse(BaseModel):
    answer: str = Field(..., description="The bot's response")
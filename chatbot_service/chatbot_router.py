from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from chatbot_service.chatbot_schema import (
    ChatRequest, ChatResponse, MemoryInfo, HealthResponse, 
    EvaluationRequest, SingleEvaluationResponse, SimpleChatRequest, SimpleChatResponse
)
from chatbot_service.chatbot import (
    memory_manager, create_memory_aware_qa_chain, get_available_models, 
    is_vectorstore_loaded, get_active_sessions_count, evaluate_groundedness,
    evaluate_relevance, calculate_detailed_evaluation_metrics
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to TeachBot with Memory API",
        "description": "Teaching Assistant with Long-Short Term Memory",
        "features": "Short-term: conversation history, Long-term: PDF knowledge base",
        "docs": "/docs",
        "health": "/health"
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    available_models = get_available_models()
    
    return HealthResponse(
        status="healthy",
        vectorstore_loaded=is_vectorstore_loaded(),
        available_models=available_models,
        active_sessions=get_active_sessions_count(),
        timestamp=datetime.now()
    )




@router.post("/simple-chat", response_model=SimpleChatResponse)
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
    



    
@router.post("/chat", response_model=ChatResponse)
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

@router.get("/memory/{session_id}", response_model=MemoryInfo)
async def get_memory_info(session_id: str):
    """Get memory information for a session"""
    session_info = memory_manager.get_session_info(session_id)
    
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return MemoryInfo(
        session_id=session_id,
        conversation_length=len(session_info.messages) // 2,
        short_term_memory=session_info.messages[-10:],  # Last 10 messages
        long_term_memory_status="loaded" if is_vectorstore_loaded() else "not loaded",
        created_at=session_info.created_at,
        last_accessed=session_info.last_accessed
    )

@router.delete("/memory/{session_id}")
async def clear_session_memory(session_id: str):
    """Clear memory for a specific session"""
    if session_id in memory_manager.sessions:
        del memory_manager.sessions[session_id]
        return {"message": f"Memory cleared for session {session_id}"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")



@router.post("/evaluate", response_model=SingleEvaluationResponse)
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
import os
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn

from chatbot_service.chatbot_router import router as chatbot_router
from chatbot_service.chatbot_schema import ErrorResponse
from chatbot_service.chatbot import load_vectorstore, memory_manager
from document_processing.document_extract_router import router as document_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory cleanup configuration
MEMORY_CLEANUP_INTERVAL = 3600  # Cleanup old sessions every hour

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

# Exception handlers with proper JSON serialization
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

# Include routers
app.include_router(chatbot_router)
app.include_router(document_router)

# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
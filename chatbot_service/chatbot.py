import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
import re
import numpy as np

from fastapi import HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv, find_dotenv

from chatbot_service.chatbot_schema import ChatMessage, ConversationHistory

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
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

# Memory configuration
MAX_CONVERSATION_HISTORY = 10  # Number of recent exchanges to keep
SESSION_TIMEOUT = 7200  # 2 hours session timeout

# Global variables
vectorstore = None  # Long-term memory
embedding_model = None

# FIXED: Simplified prompt template that works with RetrievalQA
MEMORY_AWARE_PROMPT_TEMPLATE = """
আপনি একজন সহায়ক শিক্ষা সহকারী যিনি শিক্ষার্থীদের সাথে ধারাবাহিক কথোপকথনে অংশ নেন।

নিম্নলিখিত নিয়মগুলি অনুসরণ করুন:
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
        model="gpt-4.1",
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

def get_available_models() -> List[str]:
    """Get list of available LLM models"""
    available_models = []
    
    if OPENAI_API_KEY:
        available_models.append("openai")
    if GROQ_API_KEY:
        available_models.append("groq")
    
    return available_models

def is_vectorstore_loaded() -> bool:
    """Check if vectorstore is loaded"""
    return vectorstore is not None

def get_active_sessions_count() -> int:
    """Get number of active sessions"""
    return memory_manager.get_active_sessions_count()

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    return memory_manager
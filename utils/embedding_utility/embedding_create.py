import logging
import openai
import asyncio
import re
from typing import List, Tuple, Optional
from langdetect import detect
from com.mhire.app.config.config import Config

logger = logging.getLogger(__name__)

class EmbeddingCreator:
    def __init__(self):
        self.config = Config()
        self.openai_client = openai.AsyncOpenAI(
            api_key=self.config.openai_api_key
        )
        self.model = "text-embedding-3-small"  # Efficient and good for multilingual content
        self.max_tokens = 8000  # Safe limit for the model
        
    def detect_language(self, text: str) -> Optional[str]:
        """Detect if text is Bengali, English, or mixed"""
        try:
            # Count Bengali characters
            bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            total_chars = bengali_chars + english_chars
            
            if total_chars == 0:
                return "unknown"
            
            bengali_ratio = bengali_chars / total_chars
            
            if bengali_ratio > 0.6:
                return "bengali"
            elif bengali_ratio < 0.2:
                return "english"
            else:
                return "mixed"
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown"
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better embedding quality"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Bengali and English
        text = re.sub(r'[^\u0980-\u09FF\w\s.,;:!?()-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, max_length: int = 6000) -> List[str]:
        """Split text into chunks that fit within token limits"""
        if len(text) <= max_length:
            return [text]
        
        # Try to split on sentences first
        sentences = re.split(r'[।.!?]\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If chunks are still too long, split by character count
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Split large chunks into smaller pieces
                for i in range(0, len(chunk), max_length):
                    final_chunks.append(chunk[i:i + max_length])
        
        return final_chunks
    
    async def create_embedding(self, text: str) -> Tuple[List[float], str]:
        """Create embedding for the given text"""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Detect language
            language = self.detect_language(processed_text)
            
            if not processed_text.strip():
                raise ValueError("Text is empty after preprocessing")
            
            # For very long texts, we'll use the first chunk
            # In production, you might want to create embeddings for all chunks
            chunks = self.chunk_text(processed_text)
            main_text = chunks[0]  # Use first chunk for embedding
            
            logger.info(f"Creating embedding for text of length {len(main_text)}, language: {language}")
            
            response = await self.openai_client.embeddings.create(
                input=main_text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            
            logger.info(f"Successfully created embedding with {len(embedding)} dimensions")
            
            return embedding, language
            
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[Tuple[List[float], str]]:
        """Create embeddings for multiple texts in batch"""
        try:
            # Process texts
            processed_data = []
            for text in texts:
                processed_text = self.preprocess_text(text)
                language = self.detect_language(processed_text)
                chunks = self.chunk_text(processed_text)
                processed_data.append((chunks[0], language))
            
            # Extract just the text for embedding
            texts_for_embedding = [data[0] for data in processed_data]
            
            response = await self.openai_client.embeddings.create(
                input=texts_for_embedding,
                model=self.model
            )
            
            # Combine embeddings with language info
            results = []
            for i, embedding_data in enumerate(response.data):
                embedding = embedding_data.embedding
                language = processed_data[i][1]
                results.append((embedding, language))
            
            logger.info(f"Successfully created {len(results)} embeddings")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to create batch embeddings: {e}")
            raise
    
    async def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for search query"""
        try:
            processed_query = self.preprocess_text(query)
            
            if not processed_query.strip():
                raise ValueError("Query is empty after preprocessing")
            
            response = await self.openai_client.embeddings.create(
                input=processed_query,
                model=self.model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to create query embedding: {e}")
            raise
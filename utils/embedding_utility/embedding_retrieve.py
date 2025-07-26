import logging
from typing import List, Dict, Any, Optional
from com.mhire.app.database.db_connection.db_connection import DBConnection
from com.mhire.app.utils.embedding_utility.embedding_create import EmbeddingCreator

logger = logging.getLogger(__name__)

class EmbeddingRetriever:
    def __init__(self):
        self.db_connection = DBConnection()
        self.embedding_creator = EmbeddingCreator()
        self.collection = self.db_connection.collection
        
    async def retrieve_similar_documents(
        self, 
        query_text: str, 
        limit: int = 5, 
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents using vector search for RAG implementation.
        
        Args:
            query_text: The user's query text
            limit: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            logger.info(f"Retrieving documents for query: {query_text[:100]}...")
            
            # Create query embedding
            query_embedding = await self.embedding_creator.create_query_embedding(query_text)
            
            # Vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,  # More candidates for better results
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "file_name": 1,
                        "text": 1,
                        "language_detected": 1,
                        "created_at": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "score": {"$gte": similarity_threshold}
                    }
                }
            ]
            
            results = []
            async for doc in self.collection.aggregate(pipeline):
                result = {
                    "file_name": doc["file_name"],
                    "text": doc["text"],
                    "language_detected": doc.get("language_detected", "unknown"),
                    "similarity_score": doc["score"],
                    "created_at": doc["created_at"]
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    async def retrieve_context_for_rag(
        self, 
        query_text: str, 
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """
        Retrieve and format context for RAG (Retrieval Augmented Generation).
        
        Args:
            query_text: The user's query
            max_context_length: Maximum length of combined context
            
        Returns:
            Dictionary containing formatted context and metadata
        """
        try:
            # Retrieve relevant documents
            documents = await self.retrieve_similar_documents(
                query_text=query_text,
                limit=5,
                similarity_threshold=0.6
            )
            
            if not documents:
                return {
                    "context": "",
                    "sources": [],
                    "total_documents": 0,
                    "languages_detected": []
                }
            
            # Format context
            context_parts = []
            sources = []
            languages = set()
            current_length = 0
            
            for i, doc in enumerate(documents):
                # Add document info
                doc_text = doc["text"]
                doc_info = f"Document {i+1} (Source: {doc['file_name']}):\n{doc_text}\n"
                
                # Check if adding this document would exceed max length
                if current_length + len(doc_info) > max_context_length:
                    # Truncate the text to fit
                    remaining_space = max_context_length - current_length - 50  # Leave some buffer
                    if remaining_space > 100:  # Only add if there's meaningful space
                        truncated_text = doc_text[:remaining_space] + "..."
                        doc_info = f"Document {i+1} (Source: {doc['file_name']}):\n{truncated_text}\n"
                        context_parts.append(doc_info)
                        current_length += len(doc_info)
                    break
                
                context_parts.append(doc_info)
                current_length += len(doc_info)
                
                # Collect metadata
                sources.append({
                    "file_name": doc["file_name"],
                    "similarity_score": round(doc["similarity_score"], 4),
                    "language": doc["language_detected"]
                })
                languages.add(doc["language_detected"])
            
            # Combine context
            formatted_context = "\n".join(context_parts)
            
            return {
                "context": formatted_context,
                "sources": sources,
                "total_documents": len(sources),
                "languages_detected": list(languages),
                "context_length": len(formatted_context)
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve context for RAG: {e}")
            return {
                "context": "",
                "sources": [],
                "total_documents": 0,
                "languages_detected": [],
                "error": str(e)
            }
    
    async def test_retrieval(self, query_text: str) -> Dict[str, Any]:
        """
        Test retrieval functionality - for debugging and testing purposes.
        
        Args:
            query_text: Test query
            
        Returns:
            Test results with detailed information
        """
        try:
            logger.info(f"Testing retrieval for query: {query_text}")
            
            # Get RAG context
            rag_result = await self.retrieve_context_for_rag(query_text)
            
            # Get raw documents for comparison
            raw_documents = await self.retrieve_similar_documents(query_text, limit=10)
            
            return {
                "query": query_text,
                "rag_context": rag_result,
                "raw_documents_count": len(raw_documents),
                "raw_documents": raw_documents[:3],  # First 3 for preview
                "test_status": "success"
            }
            
        except Exception as e:
            logger.error(f"Retrieval test failed: {e}")
            return {
                "query": query_text,
                "test_status": "failed",
                "error": str(e)
            }
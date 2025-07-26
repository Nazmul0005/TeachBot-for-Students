"""
RAG Evaluation System for TeachBot
Evaluates Groundedness and Relevance of the RAG system
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import requests
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question: str
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    groundedness_score: float
    relevance_score: float
    response_time: float
    model_used: str
    session_id: str
    
@dataclass
class GroundednessMetrics:
    """Groundedness evaluation metrics"""
    context_overlap_score: float  # How much answer overlaps with context
    citation_score: float  # How well answer is supported by sources
    hallucination_score: float  # Inverse of groundedness (lower is better)
    semantic_similarity: float  # Semantic similarity between answer and context

@dataclass
class RelevanceMetrics:
    """Relevance evaluation metrics"""
    query_doc_similarity: float  # Similarity between query and retrieved docs
    doc_ranking_score: float  # Quality of document ranking
    retrieval_precision: float  # Precision of retrieved documents
    coverage_score: float  # How well docs cover the query topic

class RAGEvaluator:
    """Comprehensive RAG evaluation system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip('/')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.stop_words = set(stopwords.words('english'))
        
        # Evaluation datasets
        self.test_questions = []
        self.ground_truth = {}
        
    def load_test_dataset(self, dataset_path: Optional[str] = None):
        """Load or create test dataset"""
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.test_questions = data.get('questions', [])
                self.ground_truth = data.get('ground_truth', {})
        else:
            # Create default test dataset
            self.create_default_test_dataset()
    
    def create_default_test_dataset(self):
        """Create default test questions for evaluation"""
        self.test_questions = [
            # Bengali Literature Questions
            {
                "id": "q1",
                "question": "মঞ্জরী কি?",
                "language": "bengali",
                "topic": "literature",
                "expected_keywords": ["কিশলয়", "ডাল", "মুকুল", "ফুল"]
            },
            {
                "id": "q2", 
                "question": "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?",
                "language": "bengali",
                "topic": "literature",
                "expected_keywords": ["কবি", "সাহ��ত্যিক", "নোবেল"]
            },
            # English Questions
            {
                "id": "q3",
                "question": "What is photosynthesis?",
                "language": "english", 
                "topic": "science",
                "expected_keywords": ["plant", "light", "chlorophyll", "oxygen"]
            },
            {
                "id": "q4",
                "question": "Define democracy",
                "language": "english",
                "topic": "politics",
                "expected_keywords": ["government", "people", "vote", "election"]
            },
            # Follow-up questions (for memory testing)
            {
                "id": "q5",
                "question": "এটি কোন ধরনের উদ্ভিদের অংশ?",
                "language": "bengali",
                "topic": "literature",
                "context_dependent": True,
                "depends_on": "q1"
            }
        ]
        
        # Ground truth for some questions
        self.ground_truth = {
            "q1": {
                "expected_answer": "কিশলয়যুক্ত ���চি ডাল। মুকুল",
                "relevant_docs": ["literature", "bengali_poetry"],
                "key_concepts": ["মঞ্জরী", "কিশলয়", "ডাল"]
            }
        }
    
    def query_rag_system(self, question: str, session_id: Optional[str] = None, use_memory: bool = True) -> Dict[str, Any]:
        """Query the RAG system and get response"""
        try:
            payload = {
                "message": question,
                "session_id": session_id,
                "use_memory": use_memory,
                "show_sources": True,
                "k_docs": 5
            }
            
            start_time = time.time()
            response = requests.post(f"{self.api_base_url}/chat", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                data['response_time'] = end_time - start_time
                return data
            else:
                logger.error(f"API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return None
    
    def evaluate_groundedness(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> GroundednessMetrics:
        """Evaluate how well the answer is grounded in retrieved context"""
        
        # Combine all retrieved document content
        context_text = " ".join([
            doc.get('content_preview', '') for doc in retrieved_docs
        ])
        
        if not context_text.strip():
            return GroundednessMetrics(0.0, 0.0, 1.0, 0.0)
        
        # 1. Context Overlap Score (TF-IDF based)
        context_overlap = self._calculate_context_overlap(answer, context_text)
        
        # 2. Citation Score (how well answer is supported)
        citation_score = self._calculate_citation_score(answer, retrieved_docs)
        
        # 3. Hallucination Score (inverse of groundedness)
        hallucination_score = self._calculate_hallucination_score(answer, context_text)
        
        # 4. Semantic Similarity
        semantic_similarity = self._calculate_semantic_similarity(answer, context_text)
        
        return GroundednessMetrics(
            context_overlap_score=context_overlap,
            citation_score=citation_score,
            hallucination_score=hallucination_score,
            semantic_similarity=semantic_similarity
        )
    
    def evaluate_relevance(self, question: str, retrieved_docs: List[Dict[str, Any]], expected_keywords: List[str] = None) -> RelevanceMetrics:
        """Evaluate relevance of retrieved documents to the query"""
        
        if not retrieved_docs:
            return RelevanceMetrics(0.0, 0.0, 0.0, 0.0)
        
        # 1. Query-Document Similarity
        query_doc_similarity = self._calculate_query_doc_similarity(question, retrieved_docs)
        
        # 2. Document Ranking Score
        doc_ranking_score = self._calculate_doc_ranking_score(question, retrieved_docs)
        
        # 3. Retrieval Precision
        retrieval_precision = self._calculate_retrieval_precision(retrieved_docs, expected_keywords)
        
        # 4. Coverage Score
        coverage_score = self._calculate_coverage_score(question, retrieved_docs)
        
        return RelevanceMetrics(
            query_doc_similarity=query_doc_similarity,
            doc_ranking_score=doc_ranking_score,
            retrieval_precision=retrieval_precision,
            coverage_score=coverage_score
        )
    
    def _calculate_context_overlap(self, answer: str, context: str) -> float:
        """Calculate overlap between answer and context using TF-IDF"""
        try:
            # Tokenize and clean text
            answer_words = set(word_tokenize(answer.lower())) - self.stop_words
            context_words = set(word_tokenize(context.lower())) - self.stop_words
            
            if not answer_words or not context_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = answer_words.intersection(context_words)
            union = answer_words.union(context_words)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating context overlap: {e}")
            return 0.0
    
    def _calculate_citation_score(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate how well the answer is supported by citations"""
        try:
            # Simple heuristic: check if answer contains information from multiple sources
            answer_lower = answer.lower()
            supported_docs = 0
            
            for doc in retrieved_docs:
                doc_content = doc.get('content_preview', '').lower()
                if doc_content:
                    # Check for word overlap
                    doc_words = set(word_tokenize(doc_content)) - self.stop_words
                    answer_words = set(word_tokenize(answer_lower)) - self.stop_words
                    
                    overlap = len(doc_words.intersection(answer_words))
                    if overlap > 2:  # At least 3 common words
                        supported_docs += 1
            
            return min(supported_docs / len(retrieved_docs), 1.0) if retrieved_docs else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating citation score: {e}")
            return 0.0
    
    def _calculate_hallucination_score(self, answer: str, context: str) -> float:
        """Calculate hallucination score (lower is better)"""
        try:
            # Use semantic similarity - high similarity means low hallucination
            similarity = self._calculate_semantic_similarity(answer, context)
            return 1.0 - similarity  # Invert so lower is better
            
        except Exception as e:
            logger.error(f"Error calculating hallucination score: {e}")
            return 1.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            if not text1.strip() or not text2.strip():
                return 0.0
            
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_query_doc_similarity(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate average similarity between query and retrieved documents"""
        try:
            if not retrieved_docs:
                return 0.0
            
            similarities = []
            for doc in retrieved_docs:
                doc_content = doc.get('content_preview', '')
                if doc_content:
                    similarity = self._calculate_semantic_similarity(query, doc_content)
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating query-doc similarity: {e}")
            return 0.0
    
    def _calculate_doc_ranking_score(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate quality of document ranking (DCG-like metric)"""
        try:
            if not retrieved_docs:
                return 0.0
            
            # Calculate relevance scores for each position
            dcg = 0.0
            for i, doc in enumerate(retrieved_docs):
                doc_content = doc.get('content_preview', '')
                if doc_content:
                    relevance = self._calculate_semantic_similarity(query, doc_content)
                    # DCG formula: relevance / log2(position + 2)
                    dcg += relevance / np.log2(i + 2)
            
            # Normalize by ideal DCG (assuming perfect ranking)
            ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(len(retrieved_docs)))
            
            return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating doc ranking score: {e}")
            return 0.0
    
    def _calculate_retrieval_precision(self, retrieved_docs: List[Dict[str, Any]], expected_keywords: List[str] = None) -> float:
        """Calculate precision of retrieved documents"""
        try:
            if not retrieved_docs or not expected_keywords:
                return 0.5  # Neutral score when no ground truth
            
            relevant_docs = 0
            for doc in retrieved_docs:
                doc_content = doc.get('content_preview', '').lower()
                if any(keyword.lower() in doc_content for keyword in expected_keywords):
                    relevant_docs += 1
            
            return relevant_docs / len(retrieved_docs)
            
        except Exception as e:
            logger.error(f"Error calculating retrieval precision: {e}")
            return 0.0
    
    def _calculate_coverage_score(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate how well documents cover the query topic"""
        try:
            if not retrieved_docs:
                return 0.0
            
            # Extract key terms from query
            query_words = set(word_tokenize(query.lower())) - self.stop_words
            
            # Check coverage across all documents
            covered_words = set()
            for doc in retrieved_docs:
                doc_content = doc.get('content_preview', '').lower()
                doc_words = set(word_tokenize(doc_content)) - self.stop_words
                covered_words.update(query_words.intersection(doc_words))
            
            return len(covered_words) / len(query_words) if query_words else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating coverage score: {e}")
            return 0.0
    
    def evaluate_single_question(self, question_data: Dict[str, Any], session_id: Optional[str] = None) -> EvaluationResult:
        """Evaluate a single question"""
        question = question_data['question']
        expected_keywords = question_data.get('expected_keywords', [])
        
        # Query the RAG system
        response = self.query_rag_system(question, session_id)
        
        if not response:
            return None
        
        # Extract response components
        answer = response['answer']
        retrieved_docs = response.get('sources', [])
        response_time = response.get('response_time', 0.0)
        model_used = response.get('model_used', 'unknown')
        session_id = response.get('session_id', 'unknown')
        
        # Evaluate groundedness
        groundedness_metrics = self.evaluate_groundedness(answer, retrieved_docs)
        
        # Evaluate relevance
        relevance_metrics = self.evaluate_relevance(question, retrieved_docs, expected_keywords)
        
        # Calculate overall scores
        groundedness_score = np.mean([
            groundedness_metrics.context_overlap_score,
            groundedness_metrics.citation_score,
            1.0 - groundedness_metrics.hallucination_score,  # Invert hallucination
            groundedness_metrics.semantic_similarity
        ])
        
        relevance_score = np.mean([
            relevance_metrics.query_doc_similarity,
            relevance_metrics.doc_ranking_score,
            relevance_metrics.retrieval_precision,
            relevance_metrics.coverage_score
        ])
        
        return EvaluationResult(
            question=question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            groundedness_score=groundedness_score,
            relevance_score=relevance_score,
            response_time=response_time,
            model_used=model_used,
            session_id=session_id
        )
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all test questions"""
        logger.info("Starting full RAG evaluation...")
        
        results = []
        session_id = None  # Use same session for context-dependent questions
        
        for question_data in self.test_questions:
            logger.info(f"Evaluating: {question_data['question']}")
            
            # Handle context-dependent questions
            if question_data.get('context_dependent'):
                # Use existing session for follow-up questions
                pass
            else:
                # Start new session for independent questions
                session_id = None
            
            result = self.evaluate_single_question(question_data, session_id)
            
            if result:
                results.append(result)
                if not session_id:
                    session_id = result.session_id
                
                # Log individual results
                logger.info(f"  Groundedness: {result.groundedness_score:.3f}")
                logger.info(f"  Relevance: {result.relevance_score:.3f}")
                logger.info(f"  Response time: {result.response_time:.3f}s")
            
            time.sleep(1)  # Small delay between requests
        
        # Calculate aggregate metrics
        if results:
            avg_groundedness = np.mean([r.groundedness_score for r in results])
            avg_relevance = np.mean([r.relevance_score for r in results])
            avg_response_time = np.mean([r.response_time for r in results])
            
            evaluation_summary = {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(results),
                "average_groundedness": float(avg_groundedness),
                "average_relevance": float(avg_relevance),
                "average_response_time": float(avg_response_time),
                "individual_results": [
                    {
                        "question": r.question,
                        "answer": r.answer[:200] + "..." if len(r.answer) > 200 else r.answer,
                        "groundedness_score": r.groundedness_score,
                        "relevance_score": r.relevance_score,
                        "response_time": r.response_time,
                        "model_used": r.model_used,
                        "num_sources": len(r.retrieved_docs)
                    }
                    for r in results
                ]
            }
            
            logger.info("Evaluation completed!")
            logger.info(f"Average Groundedness: {avg_groundedness:.3f}")
            logger.info(f"Average Relevance: {avg_relevance:.3f}")
            logger.info(f"Average Response Time: {avg_response_time:.3f}s")
            
            return evaluation_summary
        
        else:
            logger.error("No successful evaluations")
            return {"error": "No successful evaluations"}
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str = None):
        """Save evaluation results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_evaluation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {filename}")
        return filename

def main():
    """Main evaluation function"""
    print("🔍 RAG Evaluation System")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Load test dataset
    evaluator.load_test_dataset()
    
    print(f"Loaded {len(evaluator.test_questions)} test questions")
    
    # Check if API is available
    try:
        response = requests.get(f"{evaluator.api_base_url}/health")
        if response.status_code != 200:
            print("❌ RAG API is not available. Please start the server first.")
            return
        
        health_data = response.json()
        print(f"✅ RAG API is healthy")
        print(f"   Vectorstore loaded: {health_data['vectorstore_loaded']}")
        print(f"   Available models: {health_data['available_models']}")
        
    except Exception as e:
        print(f"❌ Could not connect to RAG API: {e}")
        return
    
    # Run evaluation
    print("\n🧪 Running evaluation...")
    results = evaluator.run_full_evaluation()
    
    if "error" not in results:
        # Save results
        filename = evaluator.save_evaluation_results(results)
        
        # Print summary
        print("\n📊 Evaluation Summary:")
        print(f"   Total Questions: {results['total_questions']}")
        print(f"   Average Groundedness: {results['average_groundedness']:.3f}")
        print(f"   Average Relevance: {results['average_relevance']:.3f}")
        print(f"   Average Response Time: {results['average_response_time']:.3f}s")
        print(f"   Results saved to: {filename}")
        
        # Show individual results
        print("\n📋 Individual Results:")
        for i, result in enumerate(results['individual_results'], 1):
            print(f"   {i}. {result['question'][:50]}...")
            print(f"      Groundedness: {result['groundedness_score']:.3f}")
            print(f"      Relevance: {result['relevance_score']:.3f}")
            print(f"      Sources: {result['num_sources']}")
            print()
    
    else:
        print(f"❌ Evaluation failed: {results['error']}")

if __name__ == "__main__":
    main()
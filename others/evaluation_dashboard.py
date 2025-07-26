"""
RAG Evaluation Dashboard
Interactive evaluation and visualization
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

class EvaluationDashboard:
    """Interactive evaluation dashboard"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
    
    def evaluate_groundedness(self, answer: str, sources: list) -> float:
        """Simple groundedness evaluation"""
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
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        answer_words = answer_words - common_words
        source_words = source_words - common_words
        
        if not answer_words:
            return 0.5
        
        overlap = len(answer_words.intersection(source_words))
        return min(overlap / len(answer_words), 1.0)
    
    def evaluate_relevance(self, question: str, sources: list) -> float:
        """Simple relevance evaluation"""
        if not sources:
            return 0.0
        
        question_words = set(re.findall(r'\w+', question.lower()))
        common_words = {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 'which', 'the', 'a', 'an'}
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
    
    def query_system(self, question: str, session_id: str = None):
        """Query the RAG system"""
        try:
            payload = {
                "message": question,
                "session_id": session_id,
                "show_sources": True,
                "k_docs": 5
            }
            
            start_time = time.time()
            response = requests.post(f"{self.api_url}/chat", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                data['response_time'] = end_time - start_time
                return data
            else:
                return None
        except Exception as e:
            st.error(f"Error: {e}")
            return None

def main():
    st.set_page_config(
        page_title="RAG Evaluation Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG Evaluation Dashboard")
    st.markdown("Evaluate Groundedness and Relevance of your TeachBot RAG system")
    
    dashboard = EvaluationDashboard()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # API health check
    try:
        health = requests.get(f"{dashboard.api_url}/health")
        if health.status_code == 200:
            health_data = health.json()
            st.sidebar.success("✅ API Connected")
            st.sidebar.info(f"Vectorstore: {'✅' if health_data['vectorstore_loaded'] else '❌'}")
            st.sidebar.info(f"Active Sessions: {health_data['active_sessions']}")
        else:
            st.sidebar.error("❌ API Not Available")
            st.error("Please start the TeachBot API server first")
            return
    except:
        st.sidebar.error("❌ Cannot Connect to API")
        st.error("Please start the TeachBot API server first")
        return
    
    # Evaluation modes
    eval_mode = st.sidebar.selectbox(
        "Evaluation Mode",
        ["Single Question", "Batch Evaluation", "Interactive Testing"]
    )
    
    if eval_mode == "Single Question":
        st.header("🔍 Single Question Evaluation")
        
        # Input question
        question = st.text_input("Enter your question:", placeholder="মঞ্জরী কি?")
        
        if st.button("Evaluate") and question:
            with st.spinner("Querying RAG system..."):
                response = dashboard.query_system(question)
            
            if response:
                answer = response['answer']
                sources = response.get('sources', [])
                response_time = response.get('response_time', 0)
                model_used = response.get('model_used', 'Unknown')
                
                # Calculate metrics
                groundedness = dashboard.evaluate_groundedness(answer, sources)
                relevance = dashboard.evaluate_relevance(question, sources)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Groundedness", f"{groundedness:.3f}")
                    if groundedness >= 0.7:
                        st.success("Good grounding")
                    elif groundedness >= 0.5:
                        st.warning("Fair grounding")
                    else:
                        st.error("Poor grounding")
                
                with col2:
                    st.metric("Relevance", f"{relevance:.3f}")
                    if relevance >= 0.7:
                        st.success("Highly relevant")
                    elif relevance >= 0.5:
                        st.warning("Moderately relevant")
                    else:
                        st.error("Low relevance")
                
                with col3:
                    st.metric("Response Time", f"{response_time:.3f}s")
                    st.info(f"Model: {model_used}")
                
                # Show answer
                st.subheader("📝 Answer")
                st.write(answer)
                
                # Show sources
                if sources:
                    st.subheader("📚 Retrieved Sources")
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"Source {i} (Page {source.get('page', 'Unknown')})"):
                            st.write(source.get('content_preview', 'No preview available'))
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[groundedness, relevance, min(response_time/5, 1)],  # Normalize response time
                    theta=['Groundedness', 'Relevance', 'Speed'],
                    fill='toself',
                    name='Evaluation Scores'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="RAG Performance Radar"
                )
                st.plotly_chart(fig)
    
    elif eval_mode == "Batch Evaluation":
        st.header("📊 Batch Evaluation")
        
        # Predefined test questions
        test_questions = [
            "মঞ্জরী কি?",
            "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?",
            "What is photosynthesis?",
            "Define democracy",
            "What is the capital of Bangladesh?"
        ]
        
        selected_questions = st.multiselect(
            "Select questions to evaluate:",
            test_questions,
            default=test_questions[:3]
        )
        
        # Add custom questions
        custom_question = st.text_input("Add custom question:")
        if custom_question and st.button("Add Question"):
            selected_questions.append(custom_question)
            st.success("Question added!")
        
        if st.button("Run Batch Evaluation") and selected_questions:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, question in enumerate(selected_questions):
                status_text.text(f"Evaluating: {question[:50]}...")
                
                response = dashboard.query_system(question)
                
                if response:
                    answer = response['answer']
                    sources = response.get('sources', [])
                    response_time = response.get('response_time', 0)
                    
                    groundedness = dashboard.evaluate_groundedness(answer, sources)
                    relevance = dashboard.evaluate_relevance(question, sources)
                    
                    results.append({
                        'Question': question,
                        'Answer': answer[:100] + "..." if len(answer) > 100 else answer,
                        'Groundedness': groundedness,
                        'Relevance': relevance,
                        'Response Time': response_time,
                        'Sources': len(sources)
                    })
                
                progress_bar.progress((i + 1) / len(selected_questions))
                time.sleep(1)
            
            status_text.text("Evaluation completed!")
            
            if results:
                # Create DataFrame
                df = pd.DataFrame(results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Groundedness", f"{df['Groundedness'].mean():.3f}")
                with col2:
                    st.metric("Avg Relevance", f"{df['Relevance'].mean():.3f}")
                with col3:
                    st.metric("Avg Response Time", f"{df['Response Time'].mean():.3f}s")
                with col4:
                    st.metric("Total Questions", len(results))
                
                # Results table
                st.subheader("📋 Detailed Results")
                st.dataframe(df)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.bar(df, x='Question', y=['Groundedness', 'Relevance'], 
                                 title="Groundedness vs Relevance by Question")
                    fig1.update_xaxis(tickangle=45)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.scatter(df, x='Groundedness', y='Relevance', 
                                     size='Response Time', hover_data=['Question'],
                                     title="Groundedness vs Relevance Scatter")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    elif eval_mode == "Interactive Testing":
        st.header("🎮 Interactive Testing")
        
        st.markdown("""
        Test the RAG system interactively and see real-time evaluation metrics.
        """)
        
        # Initialize session state
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = None
        
        # Chat interface
        question = st.text_input("Ask a question:", key="interactive_question")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Send Question"):
                if question:
                    response = dashboard.query_system(question, st.session_state.session_id)
                    
                    if response:
                        answer = response['answer']
                        sources = response.get('sources', [])
                        response_time = response.get('response_time', 0)
                        session_id = response.get('session_id')
                        
                        if not st.session_state.session_id:
                            st.session_state.session_id = session_id
                        
                        groundedness = dashboard.evaluate_groundedness(answer, sources)
                        relevance = dashboard.evaluate_relevance(question, sources)
                        
                        # Add to conversation
                        st.session_state.conversation.append({
                            'question': question,
                            'answer': answer,
                            'groundedness': groundedness,
                            'relevance': relevance,
                            'response_time': response_time,
                            'sources': len(sources)
                        })
        
        with col2:
            if st.button("Clear Conversation"):
                st.session_state.conversation = []
                st.session_state.session_id = None
                st.success("Conversation cleared!")
        
        # Display conversation
        if st.session_state.conversation:
            st.subheader("💬 Conversation History")
            
            for i, turn in enumerate(st.session_state.conversation):
                with st.expander(f"Q{i+1}: {turn['question'][:50]}..."):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Groundedness", f"{turn['groundedness']:.3f}")
                    with col2:
                        st.metric("Relevance", f"{turn['relevance']:.3f}")
                    with col3:
                        st.metric("Response Time", f"{turn['response_time']:.3f}s")
                    
                    st.write("**Answer:**", turn['answer'])
                    st.write(f"**Sources:** {turn['sources']}")
            
            # Conversation analytics
            if len(st.session_state.conversation) > 1:
                st.subheader("📈 Conversation Analytics")
                
                conv_df = pd.DataFrame(st.session_state.conversation)
                
                fig = px.line(conv_df, x=range(len(conv_df)), 
                             y=['groundedness', 'relevance'], 
                             title="Evaluation Metrics Over Time")
                fig.update_xaxis(title="Question Number")
                fig.update_yaxis(title="Score")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Enterprise GenAI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stSidebarContent"] > div:first-child {
        padding: 1.5rem 0.5rem;
    }
    div.stButton > button {
        width: 100%;
    }
    div[data-testid="stExpander"] div[data-testid="stExpanderContent"] p {
        font-size: 0.9rem;
        color: #666;
    }
    div.stChatMessage {
        padding: 0.5rem 1rem;
    }
    div.stChatMessage [data-testid="StChatMessageContent"] {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
API_URL = "http://localhost:8000/api/v1"

# Utility functions
def get_available_models() -> List[Dict]:
    """Get the list of available models from the API."""
    cache_key = "available_models"
    
    # Check if we have cached models and they're recent (less than 5 min old)
    if cache_key in st.session_state and "timestamp" in st.session_state[cache_key]:
        timestamp = st.session_state[cache_key]["timestamp"]
        if (datetime.now() - timestamp).seconds < 300:  # 5 minutes
            return st.session_state[cache_key]["data"]
    
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        
        # Cache the result
        st.session_state[cache_key] = {
            "data": models,
            "timestamp": datetime.now()
        }
        
        return models
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        # Fallback models
        return [
            {"id": "rag", "name": "RAG", "type": "rag", "description": "Retrieval Augmented Generation"},
            {"id": "bedrock", "name": "Bedrock Claude", "type": "bedrock_base", "description": "AWS Bedrock Claude model"}
        ]

def query_with_model(query: str, model_id: str, temperature: float, top_k: int = 5) -> Dict:
    """Query using the selected model."""
    try:
        with st.spinner("Processing your query..."):
            start_time = time.time()
            
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "query": query,
                    "model_id": model_id,
                    "temperature": temperature,
                    "top_k": top_k
                },
                timeout=60  # Longer timeout for complex queries
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Add processing time to the result
            result["processing_time"] = round(time.time() - start_time, 2)
            
            return result
    except requests.exceptions.Timeout:
        st.error("The request timed out. The server might be busy.")
        return {"error": "timeout"}
    except Exception as e:
        st.error(f"Error querying with model {model_id}: {str(e)}")
        return {"error": str(e)}

def submit_feedback(response_id: str, is_helpful: bool, feedback_text: str = "", 
                   query: str = "", model_id: str = "", response_text: str = "") -> bool:
    """Submit user feedback for RLHF."""
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "response_id": response_id,
                "is_helpful": is_helpful,
                "feedback": feedback_text,
                "query": query,
                "model_id": model_id,
                "response_text": response_text
            },
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        return False

def upload_document(file, metadata: Optional[Dict] = None) -> Dict:
    """Upload a document to the API for ingestion."""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {}
        
        if metadata:
            data["metadata"] = json.dumps(metadata)
        
        response = requests.post(
            f"{API_URL}/ingest",
            files=files,
            data=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return {"error": str(e)}

# UI Components
def render_sidebar():
    """Render the sidebar with model selection and settings."""
    st.sidebar.header("Model Settings")
    
    # Get available models
    models = get_available_models()
    model_options = {model["name"]: model["id"] for model in models}
    
    # Model selection
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        index=0,
        key="selected_model_name"
    )
    model_id = model_options[selected_model_name]
    
    # Store model_id in session state
    st.session_state.model_id = model_id
    
    # Get current model details
    current_model = next((model for model in models if model["id"] == model_id), None)
    
    if current_model:
        st.sidebar.markdown(f"**Model Type**: {current_model['type']}")
        if current_model.get('description'):
            st.sidebar.markdown(f"**Description**: {current_model['description']}")
    
    # Advanced settings in expander
    with st.sidebar.expander("Advanced Settings"):
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("temperature", 0.7),
            step=0.1,
            key="temperature",
            help="Higher values make output more random, lower values more deterministic"
        )
        
        # Top K documents for RAG
        top_k = st.slider(
            "Number of documents for RAG",
            min_value=1,
            max_value=20,
            value=st.session_state.get("top_k", 5),
            key="top_k",
            help="Number of relevant documents to retrieve for context"
        )
    
    # Add a clear chat button
    if st.sidebar.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    return model_id, temperature, top_k

def render_chat_interface():
    """Render the chat interface with history and input."""
    # Initialize chat history if needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show metadata if available
                metadata = {}
                if "sources" in message and message["sources"]:
                    metadata["Sources"] = message["sources"]
                if "confidence" in message:
                    metadata["Confidence"] = f"{message['confidence']:.2f}"
                if "processing_time" in message:
                    metadata["Processing time"] = f"{message['processing_time']} seconds"
                
                if metadata:
                    with st.expander("Details"):
                        for key, value in metadata.items():
                            if isinstance(value, list):
                                st.write(f"**{key}:**")
                                for item in value:
                                    st.write(f"- {item}")
                            else:
                                st.write(f"**{key}:** {value}")
                
                # Show feedback options if not already provided
                if not message.get("feedback_provided", False):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Helpful", key=f"helpful_{message.get('response_id', uuid.uuid4())}"):
                            submit_success = submit_feedback(
                                message.get("response_id", "unknown"),
                                is_helpful=True,
                                query=message.get("query", ""),
                                model_id=message.get("model_id", ""),
                                response_text=message.get("content", "")
                            )
                            if submit_success:
                                message["feedback_provided"] = True
                                st.success("Thank you for your feedback!")
                                time.sleep(1)
                                st.rerun()
                    
                    with col2:
                        if st.button("👎 Not helpful", key=f"not_helpful_{message.get('response_id', uuid.uuid4())}"):
                            feedback_text = st.text_area(
                                "What was wrong with the response?",
                                key=f"feedback_{message.get('response_id', uuid.uuid4())}"
                            )
                            if st.button("Submit Feedback", key=f"submit_{message.get('response_id', uuid.uuid4())}"):
                                submit_success = submit_feedback(
                                    message.get("response_id", "unknown"),
                                    is_helpful=False,
                                    feedback_text=feedback_text,
                                    query=message.get("query", ""),
                                    model_id=message.get("model_id", ""),
                                    response_text=message.get("content", "")
                                )
                                if submit_success:
                                    message["feedback_provided"] = True
                                    st.success("Thank you for your detailed feedback!")
                                    time.sleep(1)
                                    st.rerun()
    
    # Query input using the chat input
    query = st.chat_input("What would you like to know?")
    if query:
        process_query(query)

def process_query(query: str):
    """Process a user query and update the chat interface."""
    # Get model settings from session state
    model_id = st.session_state.get("model_id", "rag")
    temperature = st.session_state.get("temperature", 0.7)
    top_k = st.session_state.get("top_k", 5)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Display user message
    st.chat_message("user").write(query)
    
    # Generate response
    with st.chat_message("assistant"):
        result = query_with_model(query, model_id, temperature, top_k)
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
            return
        
        response_id = result.get("response_id", str(uuid.uuid4()))
        answer = result.get("answer", "Sorry, I couldn't generate a response.")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 1.0)
        processing_time = result.get("processing_time", 0)
        
        # Display answer
        st.write(answer)
        
        # Store assistant message in chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence": confidence,
            "processing_time": processing_time,
            "response_id": response_id,
            "query": query,
            "model_id": model_id,
            "feedback_provided": False
        })

def render_upload_interface():
    """Render the document upload interface."""
    st.header("Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload", 
        type=["pdf", "docx", "txt", "md", "csv", "json"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Optional metadata
        with st.expander("Add metadata"):
            metadata_text = st.text_area(
                "Enter metadata as JSON (optional)",
                placeholder='{"source": "internal", "department": "hr", "confidential": false}'
            )
            
            try:
                metadata = json.loads(metadata_text) if metadata_text.strip() else {}
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please correct the metadata.")
                metadata = {}
        
        if st.button("Upload and Process"):
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    result = upload_document(file, metadata)
                    if "error" not in result:
                        st.success(f"Successfully uploaded {file.name}")
                    else:
                        st.error(f"Error uploading {file.name}: {result['error']}")
                
                # Success message
                if len(uploaded_files) > 0:
                    st.success(f"Processed {len(uploaded_files)} documents")

def main():
    """Main application function."""
    st.title("🤖 Enterprise GenAI Assistant")
    
    # Render sidebar
    model_id, temperature, top_k = render_sidebar()
    
    # Create tabs for different functionalities
    query_tab, upload_tab = st.tabs(["Ask Questions", "Upload Documents"])
    
    with query_tab:
        render_chat_interface()
    
    with upload_tab:
        render_upload_interface()

if __name__ == "__main__":
    main() 
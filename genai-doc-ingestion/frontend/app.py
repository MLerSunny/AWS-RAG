import streamlit as st
import requests
import json
from typing import Dict, List
import uuid

# Configure page
st.set_page_config(
    page_title="Enterprise GenAI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000/api/v1"

def get_available_models() -> List[Dict]:
    """Get the list of available models from the API."""
    try:
        response = requests.get(f"{API_URL}/models")
        response.raise_for_status()
        return response.json().get("models", [])
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return [
            {"id": "rag", "name": "RAG", "type": "rag", "description": "Retrieval Augmented Generation"},
            {"id": "bedrock", "name": "Bedrock Claude", "type": "bedrock_base", "description": "AWS Bedrock Claude model"}
        ]  # Fallback models

def query_with_model(query: str, model_id: str, temperature: float, top_k: int = 5) -> Dict:
    """Query using the selected model."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "model_id": model_id,
                "temperature": temperature,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying with model {model_id}: {str(e)}")
        return {}

def submit_feedback(response_id: str, is_helpful: bool, feedback_text: str = "") -> bool:
    """Submit user feedback for RLHF."""
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "response_id": response_id,
                "is_helpful": is_helpful,
                "feedback": feedback_text
            }
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        return False

def main():
    st.title("🤖 Enterprise GenAI Assistant")
    
    # Sidebar for model selection and parameters
    st.sidebar.header("Model Settings")
    
    # Get available models
    models = get_available_models()
    model_options = {model["name"]: model["id"] for model in models}
    
    # Model selection
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        index=0
    )
    model_id = model_options[selected_model_name]
    
    # Get current model details
    current_model = next((model for model in models if model["id"] == model_id), None)
    
    if current_model:
        st.sidebar.markdown(f"**Model Type**: {current_model['type']}")
        if current_model.get('description'):
            st.sidebar.markdown(f"**Description**: {current_model['description']}")
    
    # Temperature slider
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    # Top K documents for RAG
    top_k = st.sidebar.slider(
        "Number of documents for RAG",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of relevant documents to retrieve for context"
    )
    
    # Create tabs for different functionalities
    query_tab, upload_tab = st.tabs(["Ask Questions", "Upload Documents"])
    
    with query_tab:
        # Chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.write(f"- {source}")
        
        # Query input
        query = st.chat_input("What would you like to know?")
        
        if query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Display user message
            st.chat_message("user").write(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner(f"Processing with {selected_model_name}..."):
                    # Create a placeholder for streaming effect
                    response_placeholder = st.empty()
                    
                    # Get response from API
                    result = query_with_model(query, model_id, temperature, top_k)
                    
                    if result:
                        response_id = result.get("response_id", str(uuid.uuid4()))
                        answer = result.get("answer", "Sorry, I couldn't generate a response.")
                        sources = result.get("sources", [])
                        
                        # Display answer
                        response_placeholder.write(answer)
                        
                        # Display sources if available
                        if sources:
                            with st.expander("Sources"):
                                for source in sources:
                                    st.write(f"- {source}")
                        
                        # Store assistant message in chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "response_id": response_id
                        })
                        
                        # Feedback section
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("👍 Helpful"):
                                submit_feedback(response_id, True)
                                st.success("Thank you for your feedback!")
                        
                        with col2:
                            if st.button("👎 Not helpful"):
                                feedback_text = st.text_area("What was wrong with the response?")
                                if st.button("Submit Feedback"):
                                    submit_feedback(response_id, False, feedback_text)
                                    st.success("Thank you for your detailed feedback!")
    
    with upload_tab:
        st.header("Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Choose a file to upload", 
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_file:
            if st.button("Upload and Process"):
                with st.spinner("Processing document..."):
                    # Here you would call the document ingestion API
                    st.success("Document uploaded and processed successfully!")

if __name__ == "__main__":
    main() 
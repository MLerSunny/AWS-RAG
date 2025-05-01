import streamlit as st
import requests
import json
from typing import Dict

# Configure page
st.set_page_config(
    page_title="GenAI Document Ingestion",
    page_icon="📚",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000/api/v1/query"

def query_documents(query: str, top_k: int = 5) -> Dict:
    """Query the RAG system."""
    try:
        response = requests.post(
            API_URL,
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return {}

def main():
    st.title("📚 GenAI Document Ingestion System")
    st.markdown("""
    This system allows you to query your document collection using natural language.
    The system uses RAG (Retrieval Augmented Generation) to provide accurate answers
    based on the content of your documents.
    """)
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="What would you like to know?",
        height=100
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        top_k = st.slider(
            "Number of relevant documents to retrieve:",
            min_value=1,
            max_value=10,
            value=5
        )
    
    # Query button
    if st.button("Search"):
        if query:
            with st.spinner("Searching documents..."):
                result = query_documents(query, top_k)
                
                if result:
                    # Display answer
                    st.subheader("Answer")
                    st.write(result["answer"])
                    
                    # Display sources
                    st.subheader("Sources")
                    for source in result["sources"]:
                        st.write(f"- {source}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main() 
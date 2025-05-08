# GenAI Document Ingestion System

A comprehensive system for ingesting, processing, and retrieving documents from various sources to use with Generative AI systems.

## Architecture Overview

The GenAI Document Ingestion System is designed to efficiently ingest documents from multiple sources, process them, and make them available for GenAI applications. The architecture follows a modular approach to ensure flexibility and scalability.

```
┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Document    │   │    Content    │   │   Embedding   │   │     Vector    │
│    Sources    │──▶│   Processing  │──▶│  Generation   │──▶│    Storage    │
└───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘
  SharePoint           Extraction           Embeddings         OpenSearch
  ServiceNow           Chunking             Metadata           DynamoDB
  File System          Quality Check        Processing         S3

                                    ┌───────────────┐
                                    │      API      │
                                    │    Layer      │
                                    └───────┬───────┘
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │  GenAI Query  │
                                    │     Engine    │
                                    └───────────────┘
```

### Key Components

1. **Document Sources**:
   - SharePoint Connector: Retrieves documents from SharePoint libraries
   - ServiceNow Connector: Fetches tickets and knowledge base articles
   - File System: Local and cloud-based file systems

2. **Content Processing**:
   - Document Extraction: Parse different document formats (PDF, DOCX, TXT, etc.)
   - Chunking: Divide documents into semantically meaningful chunks
   - Quality Check: Verify document quality and detect potential issues

3. **Embedding Generation**:
   - Text Embeddings: Generate vector representations of document chunks
   - Metadata Processing: Extract and enhance document metadata
   - Hallucination Detection: Check for factual consistency

4. **Vector Storage**:
   - OpenSearch: Primary vector database for semantic search
   - DynamoDB: Metadata and configuration storage
   - S3: Raw document storage and backup

5. **API Layer**:
   - REST API: Interface for external applications
   - Authentication: Secure access to the system
   - Rate Limiting: Protect against abuse

6. **GenAI Query Engine**:
   - Semantic Search: Find relevant document chunks
   - Query Augmentation: Enhance queries for better results
   - Response Generation: Create responses based on retrieved documents

## Installation

### Prerequisites

- Python 3.10 or higher
- AWS Account (for cloud deployment)
- Terraform (for infrastructure as code)

### Local Development Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/genai-doc-ingestion.git
   cd genai-doc-ingestion
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up configuration:
   ```
   cp app/config.example.py app/config.py
   # Edit app/config.py with your settings
   ```

5. Run the application:
   ```
   python run.py
   ```

### Using Docker

```
docker-compose up -d
```

## Usage Examples

### Ingesting Documents from SharePoint

```python
from genai_doc_ingestion.app.services.connectors import SharePointConnector
from genai_doc_ingestion.app.services.chunker import DocumentChunker
from genai_doc_ingestion.app.services.embedder import TextEmbedder

# Initialize components
sp_connector = SharePointConnector(
    site_url="https://example.sharepoint.com/sites/documents",
    username="your_username",
    password="your_password"
)

chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")

# List available document libraries
libraries = sp_connector.list_libraries()
print(f"Available libraries: {libraries}")

# Download and process documents
documents = sp_connector.list_documents("Technical Documents")
for doc_metadata in documents:
    # Download document
    file_obj, filename = sp_connector.download_document(doc_metadata["url"])
    
    # Extract text and chunk
    text = chunker.extract_text(file_obj, filename)
    chunks = chunker.chunk_text(text)
    
    # Generate embeddings
    for i, chunk in enumerate(chunks):
        embedding = embedder.get_embedding(chunk)
        # Store in vector database
        # ...

    print(f"Processed {filename} into {len(chunks)} chunks")
```

### Querying the System

```python
from genai_doc_ingestion.app.services.opensearch_client import OpenSearchClient
from genai_doc_ingestion.app.services.bedrock_llm import BedrockLLM

# Initialize components
search_client = OpenSearchClient(
    host="your-opensearch-endpoint.amazonaws.com",
    index_name="documents"
)
llm = BedrockLLM(model_id="anthropic.claude-v2")

# User query
query = "What is the procedure for customer onboarding?"

# Search for relevant documents
results = search_client.semantic_search(query, top_k=5)

# Format context for LLM
context = "\n\n".join([res["content"] for res in results])

# Generate response
prompt = f"""
Answer the following question using only the information provided in the context.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}
"""

response = llm.generate(prompt)
print(response)
```

## Deployment

### AWS Deployment with Terraform

The system can be deployed to AWS using the provided Terraform configurations:

1. Navigate to the terraform directory:
   ```
   cd genai-doc-ingestion/terraform
   ```

2. Initialize Terraform:
   ```
   terraform init
   ```

3. Plan the deployment:
   ```
   terraform plan -var-file=environments/dev.tfvars
   ```

4. Apply the configuration:
   ```
   terraform apply -var-file=environments/dev.tfvars
   ```

### Infrastructure Components

- **AWS Lambda**: Serverless compute for API and processing
- **Amazon OpenSearch Service**: Vector database for embeddings
- **Amazon S3**: Document storage
- **Amazon DynamoDB**: Metadata and configuration
- **Amazon API Gateway**: REST API endpoint
- **AWS IAM**: Security and access control
- **AWS CloudWatch**: Monitoring and logging
- **AWS EventBridge**: Event scheduling for batch processing

## Development

### Project Structure

```
genai-doc-ingestion/
├── app/                  # Main application code
│   ├── config.py         # Configuration
│   ├── main.py           # Application entry point
│   ├── routes/           # API routes
│   ├── services/         # Service modules
│   └── utils/            # Utility functions
├── lib/                  # Common libraries and utilities
│   ├── common/           # Shared functionality
│   └── utils/            # Utility modules
├── data/                 # Data files
├── logs/                 # Log files
├── notebooks/            # Jupyter notebooks for experimentation
├── storage/              # Local storage for development
├── terraform/            # Infrastructure as code
├── tests/                # Test suites
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── scripts/              # Utility scripts
├── .github/              # GitHub workflow configurations
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

### Testing

Run unit tests:
```
pytest tests/unit
```

Run integration tests:
```
pytest tests/integration
```

Generate coverage report:
```
pytest --cov=genai-doc-ingestion tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses several open-source libraries and tools
- Special thanks to the contributors and maintainers of those projects

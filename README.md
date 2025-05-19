# AWS RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system using AWS services.

## Features

- AWS service integration with proper error handling and retries
- Configuration management
- Utility functions for common operations
- Type hints and documentation
- Testing framework

## Prerequisites

- Python 3.8+
- AWS credentials configured
- Required AWS services:
  - S3
  - DynamoDB
  - OpenSearch

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aws-rag.git
cd aws-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and configuration
```

## Project Structure

```
aws-rag/
├── config.py           # Configuration settings
├── utils.py           # Utility functions
├── aws_base.py        # Base AWS service class
├── requirements.txt   # Project dependencies
├── tests/            # Test directory
└── README.md         # This file
```

## Usage

1. Import and use the AWS base class:
```python
from aws_base import AWSBase

class MyAWSService(AWSBase):
    def __init__(self):
        super().__init__('s3')
        
    def validate_credentials(self):
        # Implement credential validation
        pass
        
    def check_permissions(self):
        # Implement permission checks
        pass
```

2. Use utility functions:
```python
from utils import get_aws_client, load_json_file

# Get AWS client
s3_client = get_aws_client('s3')

# Load configuration
config = load_json_file('config.json')
```

## Development

1. Run tests:
```bash
pytest
```

2. Format code:
```bash
black .
```

3. Check types:
```bash
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

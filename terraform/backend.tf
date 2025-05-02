terraform {
  backend "s3" {
    bucket         = "rag-terraform-state-bucket"
    key            = "genai-doc-ingestion/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Note: The S3 bucket and DynamoDB table must be created manually before using this backend
# or they can be created via a separate Terraform configuration that uses a local backend 
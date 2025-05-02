variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Name of the S3 bucket for documents"
  type        = string
  default     = "genai-doc-ingestion-bucket"
}

variable "opensearch_domain_name" {
  description = "Name of the OpenSearch domain"
  type        = string
  default     = "genai-doc-ingestion-domain"
}

variable "opensearch_instance_type" {
  description = "Instance type for OpenSearch nodes"
  type        = string
  default     = "t3.small.search"
}

variable "opensearch_master_user" {
  description = "Master username for OpenSearch"
  type        = string
  default     = "admin"
  sensitive   = true
}

variable "opensearch_master_password" {
  description = "Master password for OpenSearch"
  type        = string
  sensitive   = true
}

variable "bedrock_model_id" {
  description = "AWS Bedrock model ID"
  type        = string
  default     = "anthropic.claude-v2"
}

variable "lambda_function_name" {
  description = "Name of the Lambda function"
  type        = string
  default     = "document-processor"
}

variable "environment" {
  description = "Deployment environment (e.g., dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vpc_id" {
  description = "VPC ID for OpenSearch domain and Lambda function"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for OpenSearch domain and Lambda function"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs for OpenSearch domain"
  type        = list(string)
} 
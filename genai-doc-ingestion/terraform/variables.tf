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

variable "lambda_function_name" {
  description = "Name of the Lambda function"
  type        = string
  default     = "document-processor"
} 
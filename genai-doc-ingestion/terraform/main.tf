terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# AWS Secrets Manager for storing credentials
resource "aws_secretsmanager_secret" "opensearch_credentials" {
  name        = "${var.environment}-opensearch-credentials"
  description = "OpenSearch credentials for the RAG application"
}

resource "aws_secretsmanager_secret_version" "opensearch_credentials" {
  secret_id     = aws_secretsmanager_secret.opensearch_credentials.id
  secret_string = jsonencode({
    username = var.opensearch_master_user
    password = var.opensearch_master_password
  })
}

resource "aws_secretsmanager_secret" "bedrock_credentials" {
  name        = "${var.environment}-bedrock-credentials"
  description = "AWS Bedrock credentials for the RAG application"
}

resource "aws_secretsmanager_secret_version" "bedrock_credentials" {
  secret_id     = aws_secretsmanager_secret.bedrock_credentials.id
  secret_string = jsonencode({
    model_id = var.bedrock_model_id
  })
}

# S3 Bucket for documents
resource "aws_s3_bucket" "documents" {
  bucket = var.bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration {
    status = "Enabled"
  }
}

# OpenSearch Domain
resource "aws_opensearch_domain" "vector_store" {
  domain_name    = var.opensearch_domain_name
  engine_version = "OpenSearch_2.11"

  cluster_config {
    instance_type = var.opensearch_instance_type
    instance_count = 1
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 10
  }

  vpc_options {
    subnet_ids         = var.subnet_ids
    security_group_ids = var.security_group_ids
  }

  node_to_node_encryption {
    enabled = true
  }

  encrypt_at_rest {
    enabled = true
  }

  advanced_security_options {
    enabled                        = true
    internal_user_database_enabled = true
    master_user_options {
      master_user_name     = var.opensearch_master_user
      master_user_password = var.opensearch_master_password
    }
  }
}

# Lambda Function
resource "aws_lambda_function" "document_processor" {
  filename         = "lambda_function.zip"
  function_name    = var.lambda_function_name
  role             = aws_iam_role.lambda_role.arn
  handler          = "lambda_handler.lambda_handler"
  runtime          = "python3.9"
  timeout          = 300
  memory_size      = 1024

  vpc_config {
    subnet_ids         = var.subnet_ids
    security_group_ids = var.security_group_ids
  }

  environment {
    variables = {
      OPENSEARCH_HOST = aws_opensearch_domain.vector_store.endpoint
      OPENSEARCH_PORT = "443"
      OPENSEARCH_CREDENTIALS_SECRET = aws_secretsmanager_secret.opensearch_credentials.name
      BEDROCK_CREDENTIALS_SECRET = aws_secretsmanager_secret.bedrock_credentials.name
      ENVIRONMENT = var.environment
    }
  }
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "document_processor_lambda_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policy for Lambda
resource "aws_iam_role_policy" "lambda_policy" {
  name = "document_processor_lambda_policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel"
        ]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.bedrock_model_id}"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.opensearch_credentials.arn,
          aws_secretsmanager_secret.bedrock_credentials.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:*:log-group:/aws/lambda/${var.lambda_function_name}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface"
        ]
        Resource = "*"
      }
    ]
  })
}

# S3 Event Notification
resource "aws_s3_bucket_notification" "document_notification" {
  bucket = aws_s3_bucket.documents.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.document_processor.arn
    events              = ["s3:ObjectCreated:*"]
  }
}

# Lambda Permission
resource "aws_lambda_permission" "s3_invoke" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.document_processor.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.documents.arn
} 

###############################
# Budget Alert for Cost Control
###############################
resource "aws_budgets_budget" "genai_monthly_budget" {
  name              = "genai-budget"
  budget_type       = "COST"
  limit_amount      = "50"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = ["Amazon Elastic Compute Cloud - Compute"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["you@example.com"]
  }
}

###############################
# IAM User for GenAI Project
###############################
resource "aws_iam_user" "genai_user" {
  name = "genai-dev-user"
  tags = {
    Environment = "dev"
    Project     = "genai-rag"
  }
}

resource "aws_iam_user_policy_attachment" "genai_user_s3_access" {
  user       = aws_iam_user.genai_user.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_user_policy_attachment" "genai_user_bedrock_access" {
  user       = aws_iam_user.genai_user.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
}

resource "aws_iam_user_policy_attachment" "genai_user_lambda_access" {
  user       = aws_iam_user.genai_user.name
  policy_arn = "arn:aws:iam::aws:policy/AWSLambda_FullAccess"
}

resource "aws_iam_user_policy_attachment" "genai_user_sagemaker_access" {
  user       = aws_iam_user.genai_user.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_user_policy_attachment" "genai_user_opensearch_access" {
  user       = aws_iam_user.genai_user.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonOpenSearchServiceFullAccess"
}
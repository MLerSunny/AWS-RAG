# Terraform Infrastructure

This directory contains the Terraform configuration for deploying the GenAI Document Ingestion infrastructure to AWS.

## Remote State Management

The Terraform state is stored in an S3 bucket with DynamoDB state locking to prevent concurrent modifications and ensure state consistency.

### Setting Up Remote State (First-Time)

The first time you set up the project, you need to create the S3 bucket and DynamoDB table for state management:

1. Comment out the `backend "s3"` block in `backend.tf` temporarily
2. Initialize Terraform with local state:

   ```bash
   terraform init
   ```

3. Apply only the state resources:

   ```bash
   terraform apply -target=aws_s3_bucket.terraform_state -target=aws_dynamodb_table.terraform_locks
   ```

4. Uncomment the `backend "s3"` block in `backend.tf`
5. Reinitialize Terraform to migrate the state to S3:

   ```bash
   terraform init -migrate-state
   ```

6. When prompted, confirm the migration of state to S3

### Regular Usage

Once the remote state is set up, you can use Terraform normally:

```bash
terraform init  # Only needed first time or when backend config changes
terraform plan  # Preview changes
terraform apply # Apply changes
```

## Resources Managed

The Terraform configuration in this directory manages the following resources:

- S3 bucket for document storage
- OpenSearch domain for vector storage
- Lambda function for document processing
- IAM roles and policies
- DynamoDB table for Terraform state locking
- S3 bucket for Terraform state

## Variables

Key variables are defined in `variables.tf` and can be overridden via command line or in a `terraform.tfvars` file.
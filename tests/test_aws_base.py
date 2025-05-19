"""Tests for AWS base class."""
import pytest
from unittest.mock import Mock, patch, call
from aws_base import AWSBase
from botocore.exceptions import ClientError
from exceptions import AWSServiceError

class TestAWSService(AWSBase):
    """Test implementation of AWS base class."""
    def validate_credentials(self) -> bool:
        return True
        
    def check_permissions(self) -> bool:
        return True

@pytest.fixture
def aws_service() -> TestAWSService:
    """Fixture for AWS service instance."""
    return TestAWSService('s3')

def test_aws_base_initialization(aws_service: TestAWSService) -> None:
    """Test AWS base class initialization."""
    assert aws_service.service_name == 's3'
    assert aws_service.client is not None
    assert aws_service.resource is not None

def test_aws_base_initialization_error() -> None:
    """Test AWS base class initialization error."""
    with patch('boto3.client', side_effect=Exception('Failed to create client')):
        with pytest.raises(AWSServiceError) as exc_info:
            TestAWSService('s3')
        assert exc_info.value.service == 's3'
        assert exc_info.value.operation == 'initialization'

def test_handle_aws_error(aws_service: TestAWSService) -> None:
    """Test AWS error handling."""
    error = ClientError(
        {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
        'GetObject'
    )
    
    with pytest.raises(AWSServiceError) as exc_info:
        aws_service.handle_aws_error(error, 'GetObject')
    
    assert exc_info.value.service == 's3'
    assert exc_info.value.operation == 'GetObject'
    assert 'Access Denied' in str(exc_info.value)

def test_get_service_status(aws_service: TestAWSService) -> None:
    """Test service status retrieval."""
    status = aws_service.get_service_status()
    
    assert status['service'] == 's3'
    assert status['status'] == 'active'

def test_get_service_status_error(aws_service: TestAWSService) -> None:
    """Test service status retrieval error."""
    with patch.object(aws_service, 'service_name', side_effect=Exception('Status check failed')):
        status = aws_service.get_service_status()
        assert status['status'] == 'error'
        assert 'Status check failed' in status['error']

@patch('time.sleep')
def test_retry_operation_success(mock_sleep: Mock, aws_service: TestAWSService) -> None:
    """Test retry operation with success."""
    mock_operation = Mock(side_effect=[
        ClientError({'Error': {'Code': 'ThrottlingException'}}, 'GetObject'),
        ClientError({'Error': {'Code': 'ThrottlingException'}}, 'GetObject'),
        'success'
    ])
    
    result = aws_service.retry_operation(mock_operation, max_retries=3)
    assert result == 'success'
    assert mock_operation.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_has_calls([call(2), call(4)])

@patch('time.sleep')
def test_retry_operation_max_retries(mock_sleep: Mock, aws_service: TestAWSService) -> None:
    """Test retry operation with maximum retries exceeded."""
    mock_operation = Mock(side_effect=[
        ClientError({'Error': {'Code': 'ThrottlingException'}}, 'GetObject'),
        ClientError({'Error': {'Code': 'ThrottlingException'}}, 'GetObject'),
        ClientError({'Error': {'Code': 'ThrottlingException'}}, 'GetObject'),
    ])
    
    with pytest.raises(AWSServiceError) as exc_info:
        aws_service.retry_operation(mock_operation, max_retries=3)
    
    assert exc_info.value.service == 's3'
    assert exc_info.value.operation == 'retry_operation'
    assert mock_operation.call_count == 3
    assert mock_sleep.call_count == 2

def test_get_resource_arn(aws_service: TestAWSService) -> None:
    """Test resource ARN generation."""
    arn = aws_service.get_resource_arn('my-bucket')
    expected_arn = f"arn:aws:s3:{aws_service.region_name}:my-bucket"
    assert arn == expected_arn

def test_validate_aws_response(aws_service: TestAWSService) -> None:
    """Test AWS response validation."""
    response = {'Key': 'value', 'Status': 'success'}
    aws_service.validate_aws_response(response, ['Key', 'Status'])

def test_validate_aws_response_missing_fields(aws_service: TestAWSService) -> None:
    """Test AWS response validation with missing fields."""
    response = {'Key': 'value'}
    
    with pytest.raises(AWSServiceError) as exc_info:
        aws_service.validate_aws_response(response, ['Key', 'Status'])
    
    assert exc_info.value.service == 's3'
    assert exc_info.value.operation == 'validate_response'
    assert 'Missing required fields' in str(exc_info.value) 
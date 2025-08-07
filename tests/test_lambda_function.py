"""
Tests for the Lambda function (agentcore_proxy.py)
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add lambda directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambda'))

# Mock boto3 before importing the lambda function
with patch('boto3.client'):
    import agentcore_proxy


class TestLambdaFunction:
    """Test cases for the Lambda function"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_context = Mock()
        self.mock_context.aws_request_id = 'test-request-123'
        
        # Mock AgentCore client
        self.mock_agent_client = Mock()
        self.mock_streaming_body = Mock()
        self.mock_streaming_body.iter_lines.return_value = [
            b'data: Hello! I am your AI assistant.',
            b'data: How can I help you today?'
        ]
        
        self.mock_agent_response = {
            'contentType': 'text/event-stream',
            'response': self.mock_streaming_body
        }
        
        self.mock_agent_client.invoke_agent_runtime.return_value = self.mock_agent_response
    
    def test_options_request(self):
        """Test CORS preflight OPTIONS request"""
        event = {
            'httpMethod': 'OPTIONS',
            'headers': {
                'Origin': 'https://example.com'
            }
        }
        
        response = agentcore_proxy.lambda_handler(event, self.mock_context)
        
        assert response['statusCode'] == 200
        assert response['body'] == ''
        assert 'Access-Control-Allow-Origin' in response['headers']
        assert response['headers']['Access-Control-Allow-Origin'] == '*'
        assert 'Access-Control-Allow-Methods' in response['headers']
        assert 'Access-Control-Allow-Headers' in response['headers']
    
    @patch('agentcore_proxy.agent_core_client')
    def test_successful_post_request(self, mock_client):
        """Test successful POST request with AgentCore response"""
        mock_client.invoke_agent_runtime.return_value = self.mock_agent_response
        
        event = {
            'httpMethod': 'POST',
            'headers': {
                'Content-Type': 'application/json',
                'Origin': 'https://example.com'
            },
            'body': json.dumps({
                'prompt': 'Hello, what can you help me with?',
                'sessionId': 'test-session-123'
            })
        }
        
        response = agentcore_proxy.lambda_handler(event, self.mock_context)
        
        assert response['statusCode'] == 200
        assert 'Access-Control-Allow-Origin' in response['headers']
        
        body = json.loads(response['body'])
        assert 'response' in body
        assert 'sessionId' in body
        assert 'timestamp' in body
        assert body['sessionId'] == 'test-session-123'
        assert body['timestamp'] == 'test-request-123'
    
    def test_missing_body(self):
        """Test request with missing body"""
        event = {
            'httpMethod': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
        response = agentcore_proxy.lambda_handler(event, self.mock_context)
        
        assert response['statusCode'] == 400
        assert 'Access-Control-Allow-Origin' in response['headers']
        
        body = json.loads(response['body'])
        assert 'error' in body
        assert 'Missing request body' in body['error']
    
    def test_invalid_json(self):
        """Test request with invalid JSON"""
        event = {
            'httpMethod': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': '{"invalid": json}'
        }
        
        response = agentcore_proxy.lambda_handler(event, self.mock_context)
        
        assert response['statusCode'] == 400
        assert 'Access-Control-Allow-Origin' in response['headers']
        
        body = json.loads(response['body'])
        assert 'error' in body
        assert 'Invalid JSON' in body['error']
    
    def test_missing_prompt(self):
        """Test request with missing prompt"""
        event = {
            'httpMethod': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'sessionId': 'test-session-123'
            })
        }
        
        response = agentcore_proxy.lambda_handler(event, self.mock_context)
        
        assert response['statusCode'] == 400
        assert 'Access-Control-Allow-Origin' in response['headers']
        
        body = json.loads(response['body'])
        assert 'error' in body
        assert 'Missing \'prompt\'' in body['error']
    
    @patch('agentcore_proxy.agent_core_client')
    def test_agentcore_error(self, mock_client):
        """Test AgentCore service error"""
        mock_client.invoke_agent_runtime.side_effect = Exception("AgentCore service unavailable")
        
        event = {
            'httpMethod': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'prompt': 'Hello',
                'sessionId': 'test-session-123'
            })
        }
        
        response = agentcore_proxy.lambda_handler(event, self.mock_context)
        
        assert response['statusCode'] == 500
        assert 'Access-Control-Allow-Origin' in response['headers']
        
        body = json.loads(response['body'])
        assert 'error' in body
        assert 'Internal server error' in body['error']
    
    def test_process_event_stream_response(self):
        """Test processing of event-stream response"""
        response_data = {
            'contentType': 'text/event-stream',
            'response': self.mock_streaming_body
        }
        
        result = agentcore_proxy.process_agentcore_response(response_data)
        
        assert 'Hello! I am your AI assistant.' in result
        assert 'How can I help you today?' in result
    
    def test_process_json_response(self):
        """Test processing of JSON response"""
        mock_streaming_body = Mock()
        mock_streaming_body.iter_lines.return_value = [
            b'{"response": "This is a JSON response"}'
        ]
        
        response_data = {
            'contentType': 'application/json',
            'response': mock_streaming_body
        }
        
        result = agentcore_proxy.process_agentcore_response(response_data)
        
        assert 'This is a JSON response' in result
    
    def test_cors_headers_in_all_responses(self):
        """Test that CORS headers are present in all response types"""
        test_cases = [
            # OPTIONS request
            {
                'httpMethod': 'OPTIONS',
                'headers': {'Origin': 'https://example.com'}
            },
            # Missing body
            {
                'httpMethod': 'POST',
                'headers': {'Content-Type': 'application/json'}
            },
            # Invalid JSON
            {
                'httpMethod': 'POST',
                'headers': {'Content-Type': 'application/json'},
                'body': 'invalid json'
            }
        ]
        
        required_cors_headers = [
            'Access-Control-Allow-Origin',
            'Access-Control-Allow-Methods',
            'Access-Control-Allow-Headers'
        ]
        
        for event in test_cases:
            response = agentcore_proxy.lambda_handler(event, self.mock_context)
            
            for header in required_cors_headers:
                assert header in response['headers'], f"Missing {header} in response for {event.get('httpMethod', 'unknown')} request"
                assert response['headers'][header] is not None
    
    @patch('agentcore_proxy.uuid.uuid4')
    @patch('agentcore_proxy.agent_core_client')
    def test_trace_id_generation(self, mock_client, mock_uuid):
        """Test that traceId is properly generated and used"""
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value='12345678-1234-1234-1234-123456789012')
        mock_client.invoke_agent_runtime.return_value = self.mock_agent_response
        
        event = {
            'httpMethod': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'prompt': 'Hello',
                'sessionId': 'test-session'
            })
        }
        
        agentcore_proxy.lambda_handler(event, self.mock_context)
        
        # Verify that invoke_agent_runtime was called with traceId
        mock_client.invoke_agent_runtime.assert_called_once()
        call_args = mock_client.invoke_agent_runtime.call_args
        assert 'traceId' in call_args.kwargs
        assert len(call_args.kwargs['traceId']) == 8  # Should be truncated to 8 characters


class TestResponseProcessing:
    """Test cases for response processing functions"""
    
    def test_create_error_response(self):
        """Test error response creation"""
        response = agentcore_proxy.create_error_response(400, "Test error message")
        
        assert response['statusCode'] == 400
        assert 'Access-Control-Allow-Origin' in response['headers']
        
        body = json.loads(response['body'])
        assert body['error'] == "Test error message"
        assert body['statusCode'] == 400
    
    def test_process_response_with_fallback(self):
        """Test response processing with fallback handling"""
        # Test with unknown response format
        response_data = {
            'contentType': 'unknown/type',
            'response': 'Simple string response'
        }
        
        result = agentcore_proxy.process_agentcore_response(response_data)
        assert 'Simple string response' in result
    
    def test_process_response_error_handling(self):
        """Test response processing error handling"""
        # Test with malformed response (None streaming body)
        response_data = {
            'contentType': 'text/event-stream',
            'response': None
        }
        
        result = agentcore_proxy.process_agentcore_response(response_data)
        # The function currently returns None for this edge case (streaming_body is None)
        # This is the actual behavior - the function doesn't handle this case explicitly
        assert result is None, f"Expected None for malformed event-stream response, got {type(result)}: '{result}'"
        
        # Test with a different content type that should reach the final fallback
        response_data_fallback = {
            'contentType': 'unknown/type',
            'response': None
        }
        
        result_fallback = agentcore_proxy.process_agentcore_response(response_data_fallback)
        # This should reach the final fallback and return str(response)
        assert isinstance(result_fallback, str), f"Expected str for fallback case, got {type(result_fallback)}"
        assert len(result_fallback) > 0, f"Expected non-empty string for fallback, got: '{result_fallback}'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
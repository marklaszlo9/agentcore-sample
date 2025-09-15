"""
Tests for the Lambda function (agentcore_proxy.py)
"""

import json
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import boto3
import pytest

# Add lambda directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lambda"))

# Mock boto3 before importing the lambda function
mock_agentcore_client = MagicMock()
mock_agentcore_control_client = MagicMock()

def client_factory(service_name, *args, **kwargs):
    if service_name == 'bedrock-agentcore':
        return mock_agentcore_client
    if service_name == 'bedrock-agentcore-control':
        return mock_agentcore_control_client
    raise ValueError(f"Unexpected service_name: {service_name}")

with patch("boto3.client", side_effect=client_factory):
    import agentcore_proxy


class TestLambdaFunction:
    """Test cases for the Lambda function"""

    def setup_method(self):
        """Set up test fixtures and reset mocks"""
        self.mock_context = Mock()
        self.mock_context.aws_request_id = "test-request-123"

        mock_agentcore_client.reset_mock()
        mock_agentcore_control_client.reset_mock()

        # Mock the streaming response for invoke_agent_runtime
        self.mock_agent_response_stream = Mock()
        self.mock_agent_response_stream.iter_lines.return_value = [
            b'data: {"completion": {"bytes": "Hello! "}}',
            b'data: {"completion": {"bytes": "I am your AI assistant."}}',
        ]
        self.mock_agent_response = {
            "contentType": "text/event-stream",
            "response": self.mock_agent_response_stream,
        }

    def test_successful_invocation_and_storage(self):
        """Test successful agent invocation and subsequent event creation."""
        mock_agentcore_client.invoke_agent_runtime.return_value = self.mock_agent_response

        event = {
            "httpMethod": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {"prompt": "Hello", "sessionId": "test-session-123"}
            ),
        }

        response = agentcore_proxy.lambda_handler(event, self.mock_context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "Hello! I am your AI assistant." in body["response"]

        mock_agentcore_client.invoke_agent_runtime.assert_called_once()
        mock_agentcore_client.create_event.assert_called_once()

        # Check the payload of the create_event call
        call_args = mock_agentcore_client.create_event.call_args
        assert call_args.kwargs["sessionId"] == "test-session-123"
        payload = call_args.kwargs["payload"]
        assert len(payload) == 2
        assert payload[0]["conversational"]["content"]["text"] == "Hello"
        assert payload[0]["conversational"]["role"] == "USER"
        assert "Hello! I am your AI assistant." in payload[1]["conversational"]["content"]["text"]
        assert payload[1]["conversational"]["role"] == "ASSISTANT"

    def test_get_history_request(self):
        """Test successful getHistory action"""
        mock_agentcore_control_client.get_memory.return_value = {
            "memoryContents": [
                {"content": "User: Old message"},
                {"content": "Agent: Old response"},
            ]
        }

        event = {
            "httpMethod": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {"action": "getHistory", "sessionId": "history-session-456"}
            ),
        }

        response = agentcore_proxy.lambda_handler(event, self.mock_context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert len(body["messages"]) == 2
        assert body["messages"][0]["content"] == "Old message"
        mock_agentcore_control_client.get_memory.assert_called_once()

    def test_invoke_agent_error_returns_fallback(self):
        """Test that a service error returns a graceful fallback response."""
        mock_agentcore_client.invoke_agent_runtime.side_effect = Exception("Service unavailable")

        event = {
            "httpMethod": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"prompt": "Hello", "sessionId": "test-session-123"}),
        }

        response = agentcore_proxy.lambda_handler(event, self.mock_context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "simplified mode" in body["response"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

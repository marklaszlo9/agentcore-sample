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
# This has to be done before the lambda_function module is imported
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

    def test_successful_post_request(self):
        """Test successful POST request for agent invocation"""
        mock_agentcore_client.invoke_agent_runtime.return_value = self.mock_agent_response
        mock_agentcore_control_client.get_memory.return_value = {"memoryContents": []}

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
        mock_agentcore_control_client.update_memory.assert_called_once()

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

    def test_store_conversation_appends_history(self):
        """Test that storing a conversation appends to existing history."""
        mock_agentcore_client.invoke_agent_runtime.return_value = self.mock_agent_response
        mock_agentcore_control_client.get_memory.return_value = {
            "memoryContents": [{"content": "User: Old prompt"}]
        }

        event = {
            "httpMethod": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {"prompt": "New prompt", "sessionId": "session-to-store"}
            ),
        }

        agentcore_proxy.lambda_handler(event, self.mock_context)

        mock_agentcore_control_client.update_memory.assert_called_once()
        call_args = mock_agentcore_control_client.update_memory.call_args
        updated_contents = call_args.kwargs["memoryContents"]
        assert len(updated_contents) == 3
        assert updated_contents[0]["content"] == "User: Old prompt"
        assert updated_contents[1]["content"] == "User: New prompt"

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

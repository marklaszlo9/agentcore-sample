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
mock_agent_core_client = MagicMock()
with patch("boto3.client", return_value=mock_agent_core_client):
    import agentcore_proxy


class TestLambdaFunction:
    """Test cases for the Lambda function"""

    def setup_method(self):
        """Set up test fixtures and reset mocks"""
        self.mock_context = Mock()
        self.mock_context.aws_request_id = "test-request-123"
        mock_agent_core_client.reset_mock()

    def test_successful_query_request(self):
        """Test a successful query request"""
        # Arrange
        mock_stream = Mock()
        mock_stream.iter_lines.return_value = [b"data: Hello there"]
        mock_agent_core_client.invoke_agent_runtime.return_value = {
            "contentType": "text/event-stream",
            "response": mock_stream,
        }

        event = {
            "httpMethod": "POST",
            "body": json.dumps({
                "prompt": "Hi",
                "sessionId": "session1"
            })
        }

        # Act
        response = agentcore_proxy.lambda_handler(event, self.mock_context)

        # Assert
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "Hello there" in body["response"]
        mock_agent_core_client.invoke_agent_runtime.assert_called_once()
        call_args = mock_agent_core_client.invoke_agent_runtime.call_args
        payload = json.loads(call_args.kwargs["payload"].decode("utf-8"))
        assert payload["prompt"] == "Hi"

    def test_successful_history_request(self):
        """Test a successful getHistory request"""
        # Arrange
        history_payload = {"messages": [{"role": "user", "content": "Old message"}]}
        mock_stream = Mock()
        mock_stream.iter_lines.return_value = [f"data: {json.dumps(history_payload)}".encode('utf-8')]
        mock_agent_core_client.invoke_agent_runtime.return_value = {
            "contentType": "text/event-stream",
            "response": mock_stream,
        }

        event = {
            "httpMethod": "POST",
            "body": json.dumps({
                "action": "getHistory",
                "sessionId": "session1"
            })
        }

        # Act
        response = agentcore_proxy.lambda_handler(event, self.mock_context)

        # Assert
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert len(body["messages"]) == 1
        assert body["messages"][0]["content"] == "Old message"
        mock_agent_core_client.invoke_agent_runtime.assert_called_once()
        call_args = mock_agent_core_client.invoke_agent_runtime.call_args
        payload = json.loads(call_args.kwargs["payload"].decode("utf-8"))
        assert payload["action"] == "getHistory"

    def test_missing_prompt_for_query(self):
        """Test that a query request with a missing prompt returns an error"""
        event = {
            "httpMethod": "POST",
            "body": json.dumps({"sessionId": "session1"})
        }
        response = agentcore_proxy.lambda_handler(event, self.mock_context)
        assert response["statusCode"] == 400
        assert "Missing 'prompt'" in response["body"]

    def test_missing_session_for_history(self):
        """Test that a history request with a missing sessionId returns an error"""
        event = {
            "httpMethod": "POST",
            "body": json.dumps({"action": "getHistory"})
        }
        response = agentcore_proxy.lambda_handler(event, self.mock_context)
        assert response["statusCode"] == 400
        assert "Missing 'sessionId'" in response["body"]

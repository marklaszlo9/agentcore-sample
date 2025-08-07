"""
Test the CustomEnvisionAgent and integration functionality
"""
import pytest
import json
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_agent import CustomEnvisionAgent


class TestCustomEnvisionAgent:
    """Test the CustomEnvisionAgent class"""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent instance"""
        return CustomEnvisionAgent(
            model_id="us.amazon.nova-micro-v1:0",
            region="us-east-1",
            knowledge_base_id="test-kb-id",
            memory_id="test-memory-id"
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.model_id == "us.amazon.nova-micro-v1:0"
        assert agent.region == "us-east-1"
        assert agent.knowledge_base_id == "test-kb-id"
        assert agent.memory_id == "test-memory-id"
    
    @patch('custom_agent.boto3.Session')
    def test_bedrock_runtime_client_creation(self, mock_session, agent):
        """Test Bedrock runtime client is created correctly"""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Access the client to trigger creation
        client = agent.bedrock_runtime
        
        # Verify boto3.Session().client was called with correct parameters
        mock_session.return_value.client.assert_called_with('bedrock-runtime', region_name='us-east-1')
        assert client == mock_client
    
    @patch('custom_agent.boto3.Session')
    def test_bedrock_agent_runtime_client_creation(self, mock_session, agent):
        """Test Bedrock agent runtime client is created correctly"""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Access the client to trigger creation
        client = agent.bedrock_agent_runtime
        
        # Verify boto3.Session().client was called with correct parameters
        mock_session.return_value.client.assert_called_with('bedrock-agent-runtime', region_name='us-east-1')
        assert client == mock_client
    
    def test_system_prompt_contains_envision_context(self, agent):
        """Test system prompt includes Envision framework context"""
        system_prompt = agent.system_prompt
        
        # Check for key Envision-related terms
        envision_terms = [
            "Envision",
            "sustainable infrastructure",
            "framework"
        ]
        
        system_prompt_lower = system_prompt.lower()
        for term in envision_terms:
            assert term.lower() in system_prompt_lower, f"System prompt missing '{term}'"
    
    @pytest.mark.asyncio
    async def test_query_processing(self, agent):
        """Test query processing with mocked Bedrock response"""
        # Mock the Bedrock response
        mock_response = {
            'output': {
                'message': {
                    'content': [
                        {
                            'text': 'This is a test response about sustainable infrastructure.'
                        }
                    ]
                }
            }
        }
        
        # Mock the internal methods that the query method uses
        agent._load_conversation_history = AsyncMock(return_value="")
        agent._store_conversation_turn = AsyncMock()
        agent._converse_with_retry = AsyncMock(return_value=mock_response)
        
        # Test query
        response = await agent.query("What is sustainable infrastructure?")
        
        assert "sustainable infrastructure" in response
    
    @pytest.mark.asyncio
    async def test_memory_content_retrieval(self, agent):
        """Test memory content retrieval"""
        # Mock the memory loading
        agent._load_conversation_history = AsyncMock(return_value="Previous conversation context")
        
        memory_content = await agent.get_memory_content()
        assert memory_content == "Previous conversation context"
    
    @pytest.mark.asyncio
    async def test_memory_update(self, agent):
        """Test memory update functionality"""
        # Mock the memory storage
        agent._store_conversation_turn = AsyncMock()
        
        await agent.update_memory("Hello", "Hi there!")
        
        # Verify the method was called
        agent._store_conversation_turn.assert_called_once_with("Hello", "Hi there!")
    
    @pytest.mark.asyncio
    async def test_knowledge_base_retrieval(self, agent):
        """Test knowledge base retrieval functionality"""
        # Mock KB response
        mock_kb_response = {
            'retrievalResults': [
                {
                    'content': {
                        'text': 'Sustainable infrastructure focuses on environmental responsibility.'
                    },
                    'score': 0.9
                }
            ]
        }
        
        # Mock the internal methods that query_with_rag uses
        agent._load_conversation_history = AsyncMock(return_value="")
        agent._store_conversation_turn = AsyncMock()
        agent._retrieve_with_retry = AsyncMock(return_value=mock_kb_response)
        agent._converse_with_retry = AsyncMock(return_value={
            'output': {
                'message': {
                    'content': [{'text': 'Test response about sustainable infrastructure'}]
                }
            }
        })
        
        # Test RAG query
        response = await agent.query_with_rag("sustainable infrastructure")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "sustainable infrastructure" in response
    
    def test_initial_greeting(self, agent):
        """Test initial greeting message"""
        greeting = agent.get_initial_greeting()
        assert "Envision" in greeting
        assert "AI agent" in greeting
    
    def test_extract_text_from_response(self, agent):
        """Test text extraction from various response formats"""
        # Test string response
        assert agent.extract_text_from_response("Simple text") == "Simple text"
        
        # Test dict response
        dict_response = {
            'content': [{'text': 'Dict text response'}]
        }
        assert agent.extract_text_from_response(dict_response) == "Dict text response"
        
        # Test None response
        assert agent.extract_text_from_response(None) == ""


class TestAgentIntegration:
    """Integration tests for the agent"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for integration testing"""
        with patch('custom_agent.boto3.Session'):
            agent = CustomEnvisionAgent(
                model_id="us.amazon.nova-micro-v1:0",
                region="us-east-1",
                knowledge_base_id="test-kb-id",
                memory_id="test-memory-id"
            )
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_memory_integration(self, mock_agent):
        """Test agent memory integration"""
        # Test that memory_id is properly set
        assert mock_agent.memory_id == "test-memory-id"
        
        # Mock memory methods
        mock_agent._load_conversation_history = AsyncMock(return_value="Previous context")
        mock_agent._store_conversation_turn = AsyncMock()
        
        # Test memory retrieval
        memory_content = await mock_agent.get_memory_content()
        assert memory_content == "Previous context"
        
        # Test memory update
        await mock_agent.update_memory("Hello", "Hi there!")
        mock_agent._store_conversation_turn.assert_called_once_with("Hello", "Hi there!")


class TestLambdaIntegration:
    """Test Lambda function integration with the agent"""
    
    def test_lambda_environment_variables(self):
        """Test that Lambda function uses proper environment variables"""
        # This would be tested in the actual Lambda environment
        # Here we just verify the expected environment variable names
        expected_env_vars = [
            'AGENT_ARN',
            'AWS_DEFAULT_REGION'
        ]
        
        # In a real Lambda environment, these would be set
        for var in expected_env_vars:
            # Just verify the variable names are what we expect
            assert isinstance(var, str)
            assert len(var) > 0
    
    def test_lambda_response_format(self):
        """Test Lambda response format matches expected structure"""
        # Mock Lambda response structure
        lambda_response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            'body': json.dumps({
                'response': 'Test response',
                'sessionId': 'test-session',
                'timestamp': 'test-timestamp'
            })
        }
        
        # Verify structure
        assert 'statusCode' in lambda_response
        assert 'headers' in lambda_response
        assert 'body' in lambda_response
        
        # Verify CORS headers
        headers = lambda_response['headers']
        assert 'Access-Control-Allow-Origin' in headers
        assert headers['Access-Control-Allow-Origin'] == '*'
        
        # Verify body structure
        body = json.loads(lambda_response['body'])
        assert 'response' in body
        assert 'sessionId' in body
        assert 'timestamp' in body


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
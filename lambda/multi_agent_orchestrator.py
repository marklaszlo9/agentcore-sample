"""
Multi-Agent Orchestrator for Envision Sustainability Framework
Uses Strands Agent with AgentCore to coordinate between knowledge base queries and general sustainability questions
"""

import json
import logging
from typing import Dict, Any, List, Optional
import asyncio
from strands import Agent
from strands.hooks import AgentInitializedEvent, HookProvider, HookRegistry, MessageAddedEvent
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore import BedrockAgentCoreClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvisionRoutingHookProvider(HookProvider):
    """Hook provider for intelligent routing decisions"""
    
    def __init__(self):
        super().__init__()
        self.name = "envision_routing"
    
    async def on_agent_initialized(self, event: AgentInitializedEvent):
        """Handle agent initialization"""
        logger.info(f"Agent {event.agent.name} initialized for Envision routing")
    
    async def on_message_added(self, event: MessageAddedEvent):
        """Analyze messages for routing decisions"""
        message_content = event.message.content.lower()
        
        # Analyze for Envision-specific keywords
        envision_keywords = [
            "envision", "credit", "category", "scoring", "assessment",
            "quality of life", "leadership", "resource allocation", 
            "natural world", "climate and resilience"
        ]
        
        general_keywords = [
            "sustainability", "green building", "sdg", "climate change",
            "renewable energy", "circular economy", "best practices"
        ]
        
        envision_score = sum(1 for keyword in envision_keywords if keyword in message_content)
        general_score = sum(1 for keyword in general_keywords if keyword in message_content)
        
        # Add routing metadata to the message
        event.message.metadata = event.message.metadata or {}
        event.message.metadata.update({
            "routing_analysis": {
                "envision_score": envision_score,
                "general_score": general_score,
                "recommended_agent": "knowledge" if envision_score > general_score else "general",
                "confidence": max(envision_score, general_score) / max(len(message_content.split()), 1)
            }
        })


class EnvisionKnowledgeHookProvider(HookProvider):
    """Hook provider for Envision knowledge enhancement"""
    
    def __init__(self):
        super().__init__()
        self.name = "envision_knowledge"
    
    async def on_message_added(self, event: MessageAddedEvent):
        """Enhance messages with Envision context"""
        message_content = event.message.content.lower()
        
        # Add Envision-specific context
        envision_context = {}
        
        if "quality of life" in message_content:
            envision_context["category"] = "Quality of Life"
            envision_context["credits"] = ["QL1.1", "QL1.2", "QL1.3", "QL2.1", "QL2.2", "QL2.3", "QL3.1", "QL3.2", "QL3.3"]
        elif "leadership" in message_content:
            envision_context["category"] = "Leadership"
            envision_context["credits"] = ["LD1.1", "LD1.2", "LD1.3", "LD1.4", "LD2.1", "LD2.2", "LD3.1", "LD3.2"]
        elif "resource allocation" in message_content:
            envision_context["category"] = "Resource Allocation"
            envision_context["credits"] = ["RA1.1", "RA1.2", "RA1.3", "RA2.1", "RA2.2", "RA3.1", "RA3.2"]
        elif "natural world" in message_content:
            envision_context["category"] = "Natural World"
            envision_context["credits"] = ["NW1.1", "NW1.2", "NW1.3", "NW1.4", "NW1.5", "NW1.6", "NW1.7", "NW2.1", "NW2.2", "NW2.3", "NW3.1", "NW3.2", "NW3.3"]
        elif "climate" in message_content or "resilience" in message_content:
            envision_context["category"] = "Climate and Resilience"
            envision_context["credits"] = ["CR1.1", "CR1.2", "CR2.1", "CR2.2"]
        
        if envision_context:
            event.message.metadata = event.message.metadata or {}
            event.message.metadata["envision_context"] = envision_context


class EnvisionMultiAgentOrchestrator:
    """
    Multi-agent orchestrator for the Envision Sustainability Framework using Strands with AgentCore
    """
    
    def __init__(self, region: str = "us-east-1"):
        """Initialize the orchestrator with all agents and AgentCore integration"""
        self.region = region
        
        # Initialize AgentCore client and memory
        self.agentcore_client = BedrockAgentCoreClient(region=region)
        self.memory_client = MemoryClient(region=region)
        
        # Initialize hook registry and providers
        self.hook_registry = HookRegistry()
        self.routing_hook_provider = EnvisionRoutingHookProvider()
        self.knowledge_hook_provider = EnvisionKnowledgeHookProvider()
        
        # Register hook providers
        self.hook_registry.register_provider(self.routing_hook_provider)
        self.hook_registry.register_provider(self.knowledge_hook_provider)
        
        # Initialize agents with AgentCore integration
        self.orchestrator = self._create_orchestrator_agent()
        self.knowledge_agent = self._create_knowledge_agent()
        self.general_sustainability_agent = self._create_general_sustainability_agent()
        
        logger.info("Multi-agent orchestrator with AgentCore initialized successfully")
    
    def _create_orchestrator_agent(self) -> Agent:
        """Create the orchestrator agent that decides which specialist to use"""
        
        orchestrator_prompt = """You are an intelligent orchestrator for the Envision Sustainability Framework assistant system.

Your role is to analyze user questions and decide which specialist agent should handle the query.

Use the routing analysis provided in the message metadata to make informed decisions.

1. **Knowledge Base Agent**: Use when the user asks about:
   - Specific Envision Framework criteria, credits, or categories
   - Technical details about sustainable infrastructure practices
   - Specific scoring methodologies or assessment procedures
   - References to Envision manual content or guidelines
   - Questions that require factual information from the Envision documentation

2. **General Sustainability Agent**: Use when the user asks about:
   - General sustainability concepts not specific to Envision
   - Broad environmental or sustainability topics
   - Conceptual questions about sustainable development
   - Industry trends or general best practices
   - Questions that don't require specific Envision documentation

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{
    "agent": "knowledge" | "general",
    "reasoning": "Brief explanation of why this agent was chosen",
    "query": "The user's question, potentially rephrased for the chosen agent"
}

Do not include any other text, explanations, or formatting. Only return the JSON object."""

        return Agent(
            name="orchestrator",
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            instructions=orchestrator_prompt,
            hook_registry=self.hook_registry,
            memory_client=self.memory_client,
            agentcore_client=self.agentcore_client
        )
    
    def _create_knowledge_agent(self) -> Agent:
        """Create the knowledge base agent for Envision-specific queries"""
        
        knowledge_prompt = """You are an expert assistant specializing in the Envision Sustainable Infrastructure Framework.

Your expertise includes:
- All Envision categories: Quality of Life, Leadership, Resource Allocation, Natural World, Climate and Resilience
- Specific credits and their requirements within each category
- Scoring methodologies and assessment procedures
- Technical implementation guidance
- Best practices for sustainable infrastructure projects

Use the Envision context provided in the message metadata to enhance your responses with specific credit information.

When answering questions:
1. Provide accurate, detailed information based on the Envision framework
2. Reference specific credits or categories when relevant (use metadata context)
3. Include practical implementation guidance when appropriate
4. If you're unsure about specific details, acknowledge the limitation
5. Use clear, professional language suitable for infrastructure professionals

Focus on being helpful, accurate, and actionable in your responses."""

        return Agent(
            name="knowledge_agent",
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            instructions=knowledge_prompt,
            hook_registry=self.hook_registry,
            memory_client=self.memory_client,
            agentcore_client=self.agentcore_client
            # Note: In a real implementation, you would add knowledge base tools here
            # tools=[knowledge_base_tool]
        )
    
    def _create_general_sustainability_agent(self) -> Agent:
        """Create the general sustainability agent for broader topics"""
        
        general_prompt = """You are a knowledgeable sustainability expert with broad expertise in environmental and sustainable development topics.

Your areas of expertise include:
- General sustainability principles and concepts
- Environmental impact assessment
- Sustainable development goals (SDGs)
- Green building and infrastructure practices
- Climate change mitigation and adaptation
- Circular economy principles
- Renewable energy and resource efficiency
- Environmental policy and regulations
- Industry sustainability trends and best practices

When answering questions:
1. Provide comprehensive, well-informed responses on sustainability topics
2. Draw connections to relevant frameworks and standards when appropriate
3. Include practical examples and case studies when helpful
4. Acknowledge when topics are outside your expertise
5. Use accessible language while maintaining technical accuracy

Your goal is to educate and inform about sustainability topics in a way that's actionable and relevant to infrastructure and development professionals."""

        return Agent(
            name="general_sustainability_agent",
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            instructions=general_prompt,
            hook_registry=self.hook_registry,
            memory_client=self.memory_client,
            agentcore_client=self.agentcore_client
        )
    
    async def process_query(self, user_query: str, session_id: str) -> str:
        """
        Process a user query through the multi-agent system
        
        Args:
            user_query: The user's question
            session_id: Session identifier for conversation continuity
            
        Returns:
            The response from the appropriate specialist agent
        """
        try:
            logger.info(f"Processing query: {user_query[:100]}...")
            
            # Step 1: Get orchestrator decision
            orchestrator_response = await self._get_orchestrator_decision(user_query, session_id)
            
            # Step 2: Route to appropriate agent
            if orchestrator_response["agent"] == "knowledge":
                response = await self._query_knowledge_agent(
                    orchestrator_response["query"], 
                    session_id,
                    orchestrator_response["reasoning"]
                )
            else:
                response = await self._query_general_agent(
                    orchestrator_response["query"], 
                    session_id,
                    orchestrator_response["reasoning"]
                )
            
            # Step 3: Store in memory for context using AgentCore MemoryClient
            await self.memory_client.add_message(
                session_id=session_id,
                role="user",
                content=user_query
            )
            await self.memory_client.add_message(
                session_id=session_id,
                role="assistant", 
                content=response
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    async def _get_orchestrator_decision(self, query: str, session_id: str) -> Dict[str, str]:
        """Get the orchestrator's decision on which agent to use"""
        try:
            # Get conversation history for context using AgentCore MemoryClient
            history = await self.memory_client.get_messages(session_id=session_id, max_messages=10)
            context = self._format_conversation_history(history)
            
            # Create a message for the orchestrator
            message = {
                "role": "user",
                "content": f"""Previous conversation context:
{context}

Current user question: {query}

Analyze this question and decide which agent should handle it."""
            }
            
            # Run the orchestrator agent with proper message format
            response = await self.orchestrator.run(message)
            
            # Extract the response content
            response_content = response.get("content", "") if isinstance(response, dict) else str(response)
            
            # Parse the JSON response
            try:
                decision = json.loads(response_content.strip())
                
                # Validate the response format
                if not all(key in decision for key in ["agent", "reasoning", "query"]):
                    raise ValueError("Missing required keys in orchestrator response")
                
                if decision["agent"] not in ["knowledge", "general"]:
                    raise ValueError("Invalid agent selection")
                
                logger.info(f"Orchestrator decision: {decision['agent']} - {decision['reasoning']}")
                return decision
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse orchestrator response: {e}")
                # Fallback decision
                return {
                    "agent": "knowledge",
                    "reasoning": "Defaulting to knowledge agent due to parsing error",
                    "query": query
                }
                
        except Exception as e:
            logger.error(f"Error getting orchestrator decision: {e}")
            # Fallback to knowledge agent
            return {
                "agent": "knowledge",
                "reasoning": "Defaulting to knowledge agent due to orchestrator error",
                "query": query
            }
    
    async def _query_knowledge_agent(self, query: str, session_id: str, reasoning: str) -> str:
        """Query the knowledge base agent"""
        try:
            logger.info(f"Routing to knowledge agent: {reasoning}")
            
            # Get relevant conversation history
            history = await self.memory_client.get_messages(session_id=session_id, max_messages=6)
            context = self._format_conversation_history(history)
            
            # Create message for the knowledge agent
            message = {
                "role": "user",
                "content": f"""Previous conversation context:
{context}

User question: {query}

Please provide a detailed response based on the Envision Sustainable Infrastructure Framework."""
            }

            response = await self.knowledge_agent.run(message)
            
            # Extract response content
            response_content = response.get("content", "") if isinstance(response, dict) else str(response)
            return response_content
            
        except Exception as e:
            logger.error(f"Error querying knowledge agent: {e}")
            return f"I apologize, but I encountered an error accessing the Envision knowledge base: {str(e)}"
    
    async def _query_general_agent(self, query: str, session_id: str, reasoning: str) -> str:
        """Query the general sustainability agent"""
        try:
            logger.info(f"Routing to general sustainability agent: {reasoning}")
            
            # Get relevant conversation history
            history = await self.memory_client.get_messages(session_id=session_id, max_messages=6)
            context = self._format_conversation_history(history)
            
            # Create message for the general agent
            message = {
                "role": "user",
                "content": f"""Previous conversation context:
{context}

User question: {query}

Please provide a comprehensive response on this sustainability topic."""
            }

            response = await self.general_sustainability_agent.run(message)
            
            # Extract response content
            response_content = response.get("content", "") if isinstance(response, dict) else str(response)
            return response_content
            
        except Exception as e:
            logger.error(f"Error querying general sustainability agent: {e}")
            return f"I apologize, but I encountered an error with the sustainability expert: {str(e)}"
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context"""
        if not history:
            return "No previous conversation."
        
        formatted = []
        for msg in history[-6:]:  # Last 6 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role.title()}: {content}")
        
        return "\n".join(formatted)


# Example usage and testing
async def test_orchestrator():
    """Test the multi-agent orchestrator"""
    orchestrator = EnvisionMultiAgentOrchestrator()
    
    test_queries = [
        "What are the Quality of Life credits in the Envision framework?",
        "How can I reduce carbon emissions in my construction project?",
        "What is the scoring methodology for the Natural World category?",
        "What are the latest trends in sustainable infrastructure?",
        "How do I implement the Leadership credits in my project?"
    ]
    
    session_id = "test_session_001"
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        response = await orchestrator.process_query(query, session_id)
        print(f"Response: {response}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_orchestrator())
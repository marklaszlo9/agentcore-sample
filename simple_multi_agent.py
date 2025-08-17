"""
Simplified Multi-Agent Orchestrator that works with basic Strands
"""

import asyncio
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from strands import Agent
    STRANDS_AVAILABLE = True
    logger.info("✅ Strands Agent available")
except ImportError as e:
    STRANDS_AVAILABLE = False
    logger.warning(f"⚠️ Strands not available: {e}")

# Simple in-memory conversation storage
class SimpleMemory:
    def __init__(self):
        self.conversations = {}
    
    def get_messages(self, session_id: str, max_messages: int = 10):
        return self.conversations.get(session_id, [])[-max_messages:]
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append({
            "role": role,
            "content": content
        })


class SimpleMultiAgentOrchestrator:
    """Simplified multi-agent orchestrator"""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.memory = SimpleMemory()
        
        if STRANDS_AVAILABLE:
            # Create agents with minimal configuration
            self.orchestrator = Agent(
                name="orchestrator",
                model="us.amazon.nova-micro-v1:0",
                instructions=self._get_orchestrator_prompt()
            )
            
            self.knowledge_agent = Agent(
                name="knowledge_agent", 
                model="us.amazon.nova-micro-v1:0",
                instructions=self._get_knowledge_prompt()
            )
            
            self.general_agent = Agent(
                name="general_agent",
                model="us.amazon.nova-micro-v1:0", 
                instructions=self._get_general_prompt()
            )
            
            logger.info("✅ Simple multi-agent system initialized")
        else:
            logger.warning("⚠️ Strands not available, multi-agent system disabled")
    
    def _get_orchestrator_prompt(self) -> str:
        return """You are an intelligent orchestrator for the Envision Sustainability Framework.

Analyze user questions and decide which specialist should respond:

- Use "knowledge" for Envision-specific questions about credits, categories, scoring
- Use "general" for broad sustainability topics, trends, best practices

Respond with JSON only:
{
    "agent": "knowledge" | "general",
    "reasoning": "Why this agent was chosen",
    "query": "The user's question"
}"""
    
    def _get_knowledge_prompt(self) -> str:
        return """You are an expert in the Envision Sustainable Infrastructure Framework.

Provide detailed responses about:
- Envision categories: Quality of Life, Leadership, Resource Allocation, Natural World, Climate and Resilience
- Specific credits and requirements
- Scoring methodologies
- Implementation guidance

Always reference specific Envision components when relevant."""
    
    def _get_general_prompt(self) -> str:
        return """You are a sustainability expert with broad knowledge.

Provide comprehensive responses on:
- General sustainability principles
- Environmental impact assessment
- Sustainable Development Goals (SDGs)
- Green building practices
- Climate change mitigation
- Industry trends and best practices

Connect topics to relevant frameworks when appropriate."""
    
    async def process_query(self, query: str, session_id: str) -> str:
        """Process a query through the multi-agent system"""
        try:
            if not STRANDS_AVAILABLE:
                return self._fallback_response(query)
            
            logger.info(f"🤖 Processing query through multi-agent system: {query[:100]}...")
            
            # Step 1: Get routing decision
            routing_response = await self.orchestrator.run(f"Route this query: {query}")
            logger.info(f"Orchestrator response: {routing_response}")
            
            # Parse routing decision
            try:
                if isinstance(routing_response, dict):
                    decision = routing_response
                else:
                    decision = json.loads(str(routing_response))
            except:
                # Fallback routing
                decision = {
                    "agent": "knowledge",
                    "reasoning": "Fallback routing",
                    "query": query
                }
            
            # Step 2: Route to chosen agent
            if decision.get("agent") == "knowledge":
                logger.info("🎯 Routing to knowledge agent")
                response = await self.knowledge_agent.run(decision.get("query", query))
            else:
                logger.info("🎯 Routing to general sustainability agent")
                response = await self.general_agent.run(decision.get("query", query))
            
            # Extract response content
            if isinstance(response, dict):
                final_response = response.get("content", str(response))
            else:
                final_response = str(response)
            
            # Store in memory
            self.memory.add_message(session_id, "user", query)
            self.memory.add_message(session_id, "assistant", final_response)
            
            logger.info(f"✅ Multi-agent response generated: {len(final_response)} chars")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in multi-agent processing: {e}")
            return self._fallback_response(query)
    
    def _fallback_response(self, query: str) -> str:
        """Provide fallback response when multi-agent system fails"""
        return f"""Thank you for your question about: "{query}"

I'm currently operating in simplified mode. Here's some guidance:

**For Envision Framework questions:**
- Quality of Life (QL): Improve community quality of life
- Leadership (LD): Provide effective leadership and commitment
- Resource Allocation (RA): Allocate resources efficiently
- Natural World (NW): Protect and restore the natural world
- Climate and Resilience (CR): Adapt to changing conditions

**For general sustainability:**
- Consider environmental, social, and economic impacts
- Implement renewable energy and efficient systems
- Follow circular economy principles
- Engage stakeholders in sustainable development

For more detailed responses, please ensure the multi-agent system is properly configured."""

    def get_conversation_history(self, session_id: str, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        return self.memory.get_messages(session_id, max_messages)


# Create a global instance
_orchestrator_instance = None

def get_simple_orchestrator(region: str = "us-east-1") -> SimpleMultiAgentOrchestrator:
    """Get or create the simple orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SimpleMultiAgentOrchestrator(region=region)
    return _orchestrator_instance


# Test function
async def test_simple_orchestrator():
    """Test the simple orchestrator"""
    orchestrator = get_simple_orchestrator()
    
    test_queries = [
        "What are the Quality of Life credits?",
        "How can I reduce carbon emissions?",
        "What is the Natural World category scoring?"
    ]
    
    session_id = "test_session"
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        response = await orchestrator.process_query(query, session_id)
        print(f"Response: {response[:200]}...")


if __name__ == "__main__":
    asyncio.run(test_simple_orchestrator())
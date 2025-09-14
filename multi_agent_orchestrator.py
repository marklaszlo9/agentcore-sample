"""
Multi-Agent Orchestrator for Envision Sustainability Framework
Uses Strands Agent with AgentCore to coordinate between knowledge base queries and general sustainability questions
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from strands import Agent
from strands.hooks import (
    AgentInitializedEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)

# Import agent logging components
from agent_logging import (
    AgentResponseLogger,
    AgentInfo,
    ResponseData,
    RoutingInfo,
    LoggingConfigManager,
    create_agent_logger,
    create_response_wrapper,
    AgentResponseWrapper
)

# Try to import AgentCore components with fallbacks and detailed logging
logger = logging.getLogger(__name__)

try:
    logger.debug("Attempting to import MemoryClient from bedrock_agentcore.memory")
    from bedrock_agentcore.memory import MemoryClient
except ImportError as e:
    logger.debug(f"Failed to import from bedrock_agentcore.memory: {e}")
    # Fallback for different package structure
    try:
        logger.debug("Attempting to import MemoryClient from bedrock_agentcore_starter_toolkit.memory")
        from bedrock_agentcore_starter_toolkit.memory import MemoryClient
    except ImportError as e2:
        logger.warning(f"Failed to import MemoryClient from all known paths: {e2}")
        MemoryClient = None

try:
    logger.debug("Attempting to import BedrockAgentCoreClient from bedrock_agentcore")
    from bedrock_agentcore import BedrockAgentCoreClient
except ImportError as e:
    logger.debug(f"Failed to import from bedrock_agentcore: {e}")
    # Try alternative imports
    try:
        logger.debug("Attempting to import BedrockAgentCoreClient from bedrock_agentcore_starter_toolkit")
        from bedrock_agentcore_starter_toolkit import BedrockAgentCoreClient
    except ImportError as e2:
        logger.debug(f"Failed to import from bedrock_agentcore_starter_toolkit: {e2}")
        try:
            logger.debug("Attempting to import BedrockAgentCoreClient from bedrock_agentcore_starter_toolkit.client")
            from bedrock_agentcore_starter_toolkit.client import BedrockAgentCoreClient
        except ImportError as e3:
            logger.warning(f"Failed to import BedrockAgentCoreClient from all known paths: {e3}")
            BedrockAgentCoreClient = None

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
            "envision",
            "credit",
            "category",
            "scoring",
            "assessment",
            "quality of life",
            "leadership",
            "resource allocation",
            "natural world",
            "climate and resilience",
        ]

        general_keywords = [
            "sustainability",
            "green building",
            "sdg",
            "climate change",
            "renewable energy",
            "circular economy",
            "best practices",
        ]

        envision_score = sum(
            1 for keyword in envision_keywords if keyword in message_content
        )
        general_score = sum(
            1 for keyword in general_keywords if keyword in message_content
        )

        # Add routing metadata to the message
        event.message.metadata = event.message.metadata or {}
        event.message.metadata.update(
            {
                "routing_analysis": {
                    "envision_score": envision_score,
                    "general_score": general_score,
                    "recommended_agent": (
                        "knowledge" if envision_score > general_score else "general"
                    ),
                    "confidence": max(envision_score, general_score)
                    / max(len(message_content.split()), 1),
                }
            }
        )


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
            envision_context["credits"] = [
                "QL1.1",
                "QL1.2",
                "QL1.3",
                "QL2.1",
                "QL2.2",
                "QL2.3",
                "QL3.1",
                "QL3.2",
                "QL3.3",
            ]
        elif "leadership" in message_content:
            envision_context["category"] = "Leadership"
            envision_context["credits"] = [
                "LD1.1",
                "LD1.2",
                "LD1.3",
                "LD1.4",
                "LD2.1",
                "LD2.2",
                "LD3.1",
                "LD3.2",
            ]
        elif "resource allocation" in message_content:
            envision_context["category"] = "Resource Allocation"
            envision_context["credits"] = [
                "RA1.1",
                "RA1.2",
                "RA1.3",
                "RA2.1",
                "RA2.2",
                "RA3.1",
                "RA3.2",
            ]
        elif "natural world" in message_content:
            envision_context["category"] = "Natural World"
            envision_context["credits"] = [
                "NW1.1",
                "NW1.2",
                "NW1.3",
                "NW1.4",
                "NW1.5",
                "NW1.6",
                "NW1.7",
                "NW2.1",
                "NW2.2",
                "NW2.3",
                "NW3.1",
                "NW3.2",
                "NW3.3",
            ]
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

    def __init__(self, region: str = "us-east-1", memory_id: Optional[str] = None, session_id: Optional[str] = None):
        """Initialize the orchestrator with all agents and AgentCore integration"""
        self.region = region
        self.memory_id = memory_id
        self.session_id = session_id
        self.actor_id = f"envision_orchestrator_{self.session_id}" if self.session_id else "default_actor"
        self.branch_name = "main"
        
        # Initialize agent response logger
        try:
            self.agent_logger = create_agent_logger()
            logger.info("Agent response logging initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize agent response logging: {e}")
            self.agent_logger = None
        
        # Initialize response wrapper for user-facing agent identification
        try:
            self.response_wrapper = create_response_wrapper()
            logger.info("Agent response wrapper initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize agent response wrapper: {e}")
            self.response_wrapper = None

        # Initialize AgentCore client and memory with fallbacks
        if BedrockAgentCoreClient:
            try:
                self.agentcore_client = BedrockAgentCoreClient(region=region)
            except Exception as e:
                logger.warning(f"Could not initialize BedrockAgentCoreClient: {e}")
                self.agentcore_client = None
        else:
            logger.warning("BedrockAgentCoreClient not available")
            self.agentcore_client = None

        if MemoryClient:
            try:
                self.memory_client = MemoryClient()
                logger.info("Initialized MemoryClient without region.")
            except Exception as e:
                logger.error(f"Failed to initialize MemoryClient: {e}")
                self.memory_client = None
        else:
            logger.warning("MemoryClient not available")
            self.memory_client = None

        # Initialize hook providers
        self.hook_providers = [
            EnvisionRoutingHookProvider(),
            EnvisionKnowledgeHookProvider(),
        ]

        # Initialize agents with available components
        self.orchestrator = self._create_orchestrator_agent()
        self.knowledge_agent = self._create_knowledge_agent()
        self.general_sustainability_agent = self._create_general_sustainability_agent()

        logger.info("Multi-agent orchestrator initialized successfully")

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

Do not include any other text, explanations, or formatting. Only return the JSON object. Keep your response concise and under 2000 characters."""

        agent_kwargs = {
            "name": "orchestrator",
            "model": "us.amazon.nova-micro-v1:0",
            "system_prompt": orchestrator_prompt,
            "hooks": self.hook_providers,
        }

        return Agent(**agent_kwargs)

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

Focus on being helpful, accurate, and actionable in your responses. Keep your response concise and under 2000 characters."""

        agent_kwargs = {
            "name": "knowledge_agent",
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "system_prompt": knowledge_prompt,
            "hooks": self.hook_providers,
        }

        return Agent(**agent_kwargs)

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

Your goal is to educate and inform about sustainability topics in a way that's actionable and relevant to infrastructure and development professionals. Keep your response concise and under 2000 characters."""

        agent_kwargs = {
            "name": "general_sustainability_agent",
            "model": "us.anthropic.claude-opus-4-20250514-v1:0",
            "system_prompt": general_prompt,
            "hooks": self.hook_providers,
        }

        return Agent(**agent_kwargs)

    async def process_query(self, user_query: str, session_id: str) -> str:
        """
        Process a user query through the multi-agent system

        Args:
            user_query: The user's question
            session_id: Session identifier for conversation continuity

        Returns:
            The response from the appropriate specialist agent
        """
        start_time = time.time()
        orchestrator_start_time = start_time
        
        try:
            logger.info(f"Processing query: {user_query[:100]}...")

            # Step 1: Get orchestrator decision with timing
            orchestrator_response = await self._get_orchestrator_decision(
                user_query, session_id
            )
            orchestrator_end_time = time.time()
            orchestrator_execution_time = int((orchestrator_end_time - orchestrator_start_time) * 1000)

            # Step 2: Log the routing decision with model info
            chosen_agent_name = orchestrator_response.get("agent")
            reasoning = orchestrator_response.get("reasoning", "No reasoning provided.")

            if chosen_agent_name == "knowledge":
                chosen_agent_instance = self.knowledge_agent
            else:
                # Default to general agent if not knowledge
                chosen_agent_name = "general"
                chosen_agent_instance = self.general_sustainability_agent

            logger.info(
                f"Routing to agent: {chosen_agent_instance.name}, "
                f"Model: {chosen_agent_instance.model}, "
                f"Reasoning: {reasoning}"
            )

            # Log orchestrator decision
            if self.agent_logger:
                try:
                    orchestrator_info = AgentInfo(
                        agent_name="orchestrator",
                        agent_type="orchestrator",
                        model_id=self.orchestrator.model,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=orchestrator_execution_time
                    )
                    
                    routing_info = RoutingInfo(
                        selected_agent=chosen_agent_name,
                        routing_reasoning=reasoning,
                        available_agents=["knowledge", "general"]
                    )
                    
                    self.agent_logger.log_routing_decision(orchestrator_info, routing_info)
                except Exception as e:
                    logger.warning(f"Failed to log routing decision: {e}")

            # Step 3: Route to appropriate agent with timing
            agent_start_time = time.time()
            
            if chosen_agent_name == "knowledge":
                response = await self._query_knowledge_agent(
                    orchestrator_response["query"],
                    session_id
                )
            else:
                response = await self._query_general_agent(
                    orchestrator_response["query"],
                    session_id
                )
            
            agent_end_time = time.time()
            agent_execution_time = int((agent_end_time - agent_start_time) * 1000)
            total_execution_time = int((agent_end_time - start_time) * 1000)

            # Log agent response
            if self.agent_logger:
                try:
                    agent_info = AgentInfo(
                        agent_name=chosen_agent_instance.name,
                        agent_type=chosen_agent_name,
                        model_id=chosen_agent_instance.model,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=agent_execution_time
                    )
                    
                    response_data = ResponseData(
                        query=user_query,
                        response=response,
                        response_length=len(response),
                        success=True,
                        metadata={
                            "orchestrator_execution_time_ms": orchestrator_execution_time,
                            "agent_execution_time_ms": agent_execution_time,
                            "total_execution_time_ms": total_execution_time,
                            "routing_reasoning": reasoning,
                            "selected_from_agents": ["knowledge", "general"]
                        }
                    )
                    
                    self.agent_logger.log_agent_response(agent_info, response_data)
                except Exception as e:
                    logger.warning(f"Failed to log agent response: {e}")

            # Step 4: Apply user-facing response wrapper if enabled
            final_response = response
            if self.response_wrapper:
                try:
                    # Create agent info for the selected agent
                    selected_agent_info = AgentInfo(
                        agent_name=chosen_agent_instance.name,
                        agent_type=chosen_agent_name,
                        model_id=chosen_agent_instance.model,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=agent_execution_time
                    )
                    
                    # Create orchestrator info
                    orchestrator_info = AgentInfo(
                        agent_name="orchestrator",
                        agent_type="orchestrator",
                        model_id=self.orchestrator.model,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=orchestrator_execution_time
                    )
                    
                    # Wrap the response with agent identification
                    final_response = self.response_wrapper.wrap_multi_agent_response(
                        response, orchestrator_info, selected_agent_info, reasoning
                    )
                except Exception as e:
                    logger.warning(f"Failed to wrap response with agent identification: {e}")
                    final_response = response

            # Step 5: Store in memory for context
            if self.memory_client and self.memory_id:
                try:
                    await self.memory_client.create_event(
                        memory_id=self.memory_id,
                        actor_id=self.actor_id,
                        session_id=session_id,
                        messages=[
                            (user_query, "user"),
                            (final_response, "assistant"),
                        ],
                    )
                except Exception as e:
                    logger.warning(f"Could not store conversation in memory: {e}")

            return final_response

        except Exception as e:
            # Log error with timing information
            error_time = time.time()
            error_execution_time = int((error_time - start_time) * 1000)
            
            if self.agent_logger:
                try:
                    error_agent_info = AgentInfo(
                        agent_name="orchestrator",
                        agent_type="orchestrator", 
                        model_id=self.orchestrator.model,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=error_execution_time
                    )
                    
                    self.agent_logger.log_agent_error(
                        error_agent_info, 
                        e, 
                        context={
                            "query": user_query[:200],  # Truncated for privacy
                            "processing_stage": "multi_agent_orchestration"
                        }
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log agent error: {log_error}")
            
            logger.error(f"Error processing query: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

    async def _get_orchestrator_decision(
        self, query: str, session_id: str
    ) -> Dict[str, str]:
        """Get the orchestrator's decision on which agent to use"""
        try:
            # Get conversation history for context
            if self.memory_client and self.memory_id:
                try:
                    history = await self.memory_client.get_last_k_turns(
                        memory_id=self.memory_id,
                        actor_id=self.actor_id,
                        session_id=session_id,
                        k=10,
                        branch_name=self.branch_name,
                    )
                    context = self._format_conversation_history(history)
                except Exception as e:
                    logger.warning(f"Could not get conversation history: {e}")
                    context = "No previous conversation."
            else:
                context = "No previous conversation."

            # Create a message for the orchestrator
            message_content = f"""Previous conversation context:
{context}

Current user question: {query}

Analyze this question and decide which agent should handle it."""

            # Run the orchestrator agent with proper message format
            response = self.orchestrator(message_content)

            # Extract the response content
            response_content = (
                response.get("content", "")
                if isinstance(response, dict)
                else str(response)
            )

            # Parse the JSON response
            try:
                decision = json.loads(response_content.strip())

                # Validate the response format
                if not all(key in decision for key in ["agent", "reasoning", "query"]):
                    raise ValueError("Missing required keys in orchestrator response")

                if decision["agent"] not in ["knowledge", "general"]:
                    raise ValueError("Invalid agent selection")

                logger.info(
                    f"Orchestrator decision: {decision['agent']} - {decision['reasoning']}"
                )
                return decision

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse orchestrator response: {e}")
                # Fallback decision
                return {
                    "agent": "knowledge",
                    "reasoning": "Defaulting to knowledge agent due to parsing error",
                    "query": query,
                }

        except Exception as e:
            logger.error(f"Error getting orchestrator decision: {e}")
            # Fallback to knowledge agent
            return {
                "agent": "knowledge",
                "reasoning": "Defaulting to knowledge agent due to orchestrator error",
                "query": query,
            }

    async def _query_knowledge_agent(
        self, query: str, session_id: str
    ) -> str:
        """Query the knowledge base agent"""
        start_time = time.time()
        
        try:
            # Get relevant conversation history
            if self.memory_client and self.memory_id:
                try:
                    history = await self.memory_client.get_last_k_turns(
                        memory_id=self.memory_id,
                        actor_id=self.actor_id,
                        session_id=session_id,
                        k=6,
                        branch_name=self.branch_name,
                    )
                    context = self._format_conversation_history(history)
                except Exception as e:
                    logger.warning(f"Could not get conversation history: {e}")
                    context = "No previous conversation."
            else:
                context = "No previous conversation."

            # Create message for the knowledge agent
            message_content = f"""Previous conversation context:
{context}

User question: {query}

Please provide a detailed response based on the Envision Sustainable Infrastructure Framework."""

            response = self.knowledge_agent(message_content)

            # Extract response content
            response_content = (
                response.get("content", "")
                if isinstance(response, dict)
                else str(response)
            )
            
            end_time = time.time()
            execution_time = int((end_time - start_time) * 1000)
            
            # Log individual agent response
            if self.agent_logger:
                try:
                    agent_info = AgentInfo(
                        agent_name=self.knowledge_agent.name,
                        agent_type="knowledge",
                        model_id=self.knowledge_agent.model,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    response_data = ResponseData(
                        query=query,
                        response=response_content,
                        response_length=len(response_content),
                        success=True,
                        metadata={
                            "knowledge_base_used": True,
                            "context_length": len(context),
                            "agent_mode": "knowledge_specialist"
                        }
                    )
                    
                    self.agent_logger.log_agent_response(agent_info, response_data)
                except Exception as e:
                    logger.warning(f"Failed to log knowledge agent response: {e}")
            
            return response_content

        except Exception as e:
            end_time = time.time()
            execution_time = int((end_time - start_time) * 1000)
            
            # Log knowledge agent error
            if self.agent_logger:
                try:
                    error_agent_info = AgentInfo(
                        agent_name=self.knowledge_agent.name,
                        agent_type="knowledge",
                        model_id=self.knowledge_agent.model,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    self.agent_logger.log_agent_error(
                        error_agent_info,
                        e,
                        context={
                            "query": query[:200],  # Truncated for privacy
                            "agent_mode": "knowledge_specialist"
                        }
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log knowledge agent error: {log_error}")
            
            logger.error(f"Error querying knowledge agent: {e}")
            return f"I apologize, but I encountered an error accessing the Envision knowledge base: {str(e)}"

    async def _query_general_agent(
        self, query: str, session_id: str
    ) -> str:
        """Query the general sustainability agent"""
        start_time = time.time()
        
        try:
            # Get relevant conversation history
            if self.memory_client and self.memory_id:
                try:
                    history = await self.memory_client.get_last_k_turns(
                        memory_id=self.memory_id,
                        actor_id=self.actor_id,
                        session_id=session_id,
                        k=6,
                        branch_name=self.branch_name,
                    )
                    context = self._format_conversation_history(history)
                except Exception as e:
                    logger.warning(f"Could not get conversation history: {e}")
                    context = "No previous conversation."
            else:
                context = "No previous conversation."

            # Create message for the general agent
            message_content = f"""Previous conversation context:
{context}

User question: {query}

Please provide a comprehensive response on this sustainability topic."""

            response = self.general_sustainability_agent(message_content)

            # Extract response content
            response_content = (
                response.get("content", "")
                if isinstance(response, dict)
                else str(response)
            )
            
            end_time = time.time()
            execution_time = int((end_time - start_time) * 1000)
            
            # Log individual agent response
            if self.agent_logger:
                try:
                    agent_info = AgentInfo(
                        agent_name=self.general_sustainability_agent.name,
                        agent_type="general",
                        model_id=self.general_sustainability_agent.model,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    response_data = ResponseData(
                        query=query,
                        response=response_content,
                        response_length=len(response_content),
                        success=True,
                        metadata={
                            "knowledge_base_used": False,
                            "context_length": len(context),
                            "agent_mode": "general_sustainability"
                        }
                    )
                    
                    self.agent_logger.log_agent_response(agent_info, response_data)
                except Exception as e:
                    logger.warning(f"Failed to log general agent response: {e}")
            
            return response_content

        except Exception as e:
            end_time = time.time()
            execution_time = int((end_time - start_time) * 1000)
            
            # Log general agent error
            if self.agent_logger:
                try:
                    error_agent_info = AgentInfo(
                        agent_name=self.general_sustainability_agent.name,
                        agent_type="general",
                        model_id=self.general_sustainability_agent.model,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    self.agent_logger.log_agent_error(
                        error_agent_info,
                        e,
                        context={
                            "query": query[:200],  # Truncated for privacy
                            "agent_mode": "general_sustainability"
                        }
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log general agent error: {log_error}")
            
            logger.error(f"Error querying general sustainability agent: {e}")
            return f"I apologize, but I encountered an error with the sustainability expert: {str(e)}"

    def _format_conversation_history(self, history: List[List[Dict[str, Any]]]) -> str:
        """Format conversation history for context"""
        if not history:
            return "No previous conversation."

        formatted = []
        for turn in history:
            for message in turn:
                role = message.get("role", "unknown").lower()
                content = message.get("content", {}).get("text", "")
                formatted.append(f"{role.title()}: {content}")

        return "\n".join(formatted)

    async def get_history(self, session_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get conversation history from memory."""
        if not self.memory_client or not self.memory_id:
            logger.warning("MemoryClient not available, cannot retrieve history.")
            return []
        try:
            history = await self.memory_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=session_id,
                k=k,
                branch_name=self.branch_name,
            )
            return history
        except Exception as e:
            logger.error(f"Failed to retrieve history for session {session_id}: {e}")
            return []


# Example usage and testing
async def test_orchestrator():
    """Test the multi-agent orchestrator"""
    import os
    session_id = "test_session_001"
    memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    orchestrator = EnvisionMultiAgentOrchestrator(memory_id=memory_id, session_id=session_id)

    test_queries = [
        "What are the Quality of Life credits in the Envision framework?",
        "How can I reduce carbon emissions in my construction project?",
        "What is the scoring methodology for the Natural World category?",
        "What are the latest trends in sustainable infrastructure?",
        "How do I implement the Leadership credits in my project?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        response = await orchestrator.process_query(query, session_id)
        print(f"Response: {response}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_orchestrator())

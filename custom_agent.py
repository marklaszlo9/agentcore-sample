"""
Custom Agent implementation using AgentCore Memory
This preserves the existing logic while implementing proper AgentCore memory management.
Based on: https://github.com/awslabs/amazon-bedrock-agentcore-samples/tree/main/01-tutorials/04-AgentCore-memory
"""

import asyncio
import functools
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

# Import agent logging components
from agent_logging import (
    AgentResponseLogger,
    AgentInfo,
    ResponseData,
    create_agent_logger,
    create_response_wrapper,
    AgentResponseWrapper
)

logger = logging.getLogger(__name__)


# --- Simplified AgentCore MemoryClient Import ---
# Try to import MemoryClient from the most likely locations, with clear logging.
try:
    # Primary, recommended import path
    from bedrock_agentcore_starter_toolkit.memory import MemoryClient

    AGENTCORE_AVAILABLE = True
    logger.info(
        "✅ Successfully imported MemoryClient from bedrock_agentcore_starter_toolkit"
    )
except ImportError:
    try:
        # Fallback for other possible package structures
        from bedrock_agentcore.memory import MemoryClient

        AGENTCORE_AVAILABLE = True
        logger.info("✅ Successfully imported MemoryClient from bedrock_agentcore")
    except ImportError:
        MemoryClient = None
        AGENTCORE_AVAILABLE = False
        logger.warning(
            "AgentCore MemoryClient not found. Will use boto3 bedrock-agentcore fallback."
        )


# --- Decorator for AWS Credential Expiration Retry ---
def aws_retry_on_expiration(max_retries=2):
    """
    A decorator to handle AWS credential expiration by refreshing clients and retrying.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code in [
                        "ExpiredTokenException",
                        "InvalidTokenException",
                        "TokenRefreshRequired",
                    ]:
                        logger.warning(
                            f"AWS credentials expired on attempt {attempt + 1}/{max_retries + 1} "
                            f"for {func.__name__}: {e}"
                        )
                        if attempt < max_retries:
                            self.refresh_clients()
                            await asyncio.sleep(1)  # Brief delay before retry
                        else:
                            logger.error(
                                f"Max retries exceeded for credential refresh in {func.__name__}"
                            )
                            raise
                    else:
                        logger.error(
                            f"Non-credential ClientError in {func.__name__}: {e}"
                        )
                        raise
                except Exception as e:
                    logger.error(
                        f"Unexpected error in {func.__name__} (attempt {attempt + 1}): {e}"
                    )
                    if attempt >= max_retries:
                        raise
                    await asyncio.sleep(1)

        return wrapper

    return decorator


class CustomEnvisionAgent:
    """
    Custom Agent with AgentCore memory management and proper credential handling.
    Implements AgentCore memory patterns as documented in AWS docs.
    """

    def __init__(
        self,
        model_id: str = "us.amazon.nova-micro-v1:0",
        region: str = "us-east-1",
        knowledge_base_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
    ):

        # Default system prompt for Envision
        self.system_prompt = (
            system_prompt
            or """You are an expert assistant on the Envision Sustainable Infrastructure Framework Version 3. Your sole purpose is to answer questions based on the content of the provided 'ISI Envision.pdf' manual.

Follow these instructions precisely:
1.  When a user asks a question, find the answer *only* within the provided knowledge base context from the Envision manual.
2.  Provide clear, accurate, and concise answers based strictly on the information found in the document. You may quote or paraphrase from the text.
3.  If the user's question cannot be answered using the Envision manual, you must state that you can only answer questions about the Envision Sustainable Infrastructure Framework. Do not use any external knowledge or make assumptions.
4.  If the query is conversational (e.g., "hello", "thank you"), you may respond politely but briefly.
"""
        )

        self.model_id = model_id
        self.region = region
        self.knowledge_base_id = knowledge_base_id
        self.user_id = user_id or f"user_{os.urandom(8).hex()}"

        # AgentCore memory configuration
        self.memory_id = memory_id or os.environ.get("AGENTCORE_MEMORY_ID")
        self.actor_id = f"envision_agent_{self.user_id}"
        self.session_id = self.user_id
        self.branch_name = "main"
        
        # Agent identification for logging
        self.agent_name = agent_name or "custom_envision_agent"
        self.agent_type = agent_type or "custom"
        
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

        # Initialize AgentCore MemoryClient if available, otherwise use boto3 fallback
        self.memory_client = None
        self._bedrock_agentcore = None

        if AGENTCORE_AVAILABLE and self.memory_id:
            try:
                # Initialize MemoryClient with explicit region to avoid us-west-2 default
                try:
                    # Try with region parameter first to ensure correct region
                    self.memory_client = MemoryClient(region=self.region)
                    logger.info(
                        f"✅ AgentCore MemoryClient initialized with region {self.region}"
                    )
                except TypeError:
                    # If region parameter not supported, try without but set AWS_DEFAULT_REGION
                    os.environ["AWS_DEFAULT_REGION"] = self.region
                    self.memory_client = MemoryClient()
                    logger.info(
                        f"✅ AgentCore MemoryClient initialized (region set via env: {self.region})"
                    )
                except Exception as e2:
                    logger.error(f"MemoryClient initialization failed: {str(e2)}")
                    raise e2

                logger.info(
                    f"✅ AgentCore MemoryClient ready for memory_id: {self.memory_id} in region {self.region}"
                )
            except Exception as e:
                logger.warning(f"Could not initialize AgentCore MemoryClient: {str(e)}")
                # Fall back to boto3 client
                self._init_boto3_fallback()
        elif self.memory_id:
            # Use boto3 fallback when AgentCore package not available
            logger.info("Using boto3 bedrock-agentcore client fallback")
            self._init_boto3_fallback()

        # Don't initialize clients here - create them lazily to handle credential refresh
        self._bedrock_runtime = None
        self._bedrock_agent_runtime = None

        logger.info(
            f"CustomEnvisionAgent initialized with model {model_id}, region {region}, KB: {knowledge_base_id}, user: {self.user_id}, memory_id: {self.memory_id}"
        )

    @property
    def bedrock_runtime(self):
        """Lazy initialization of bedrock-runtime client to handle credential refresh"""
        if self._bedrock_runtime is None:
            try:
                # Create a new session to get fresh credentials
                session = boto3.Session()
                self._bedrock_runtime = session.client(
                    "bedrock-runtime", region_name=self.region
                )
                logger.debug("Created new bedrock-runtime client")
            except Exception as e:
                logger.error(f"Error creating bedrock-runtime client: {str(e)}")
                raise
        return self._bedrock_runtime

    @property
    def bedrock_agent_runtime(self):
        """Lazy initialization of bedrock-agent-runtime client to handle credential refresh"""
        if self._bedrock_agent_runtime is None:
            try:
                # Create a new session to get fresh credentials
                session = boto3.Session()
                self._bedrock_agent_runtime = session.client(
                    "bedrock-agent-runtime", region_name=self.region
                )
                logger.debug("Created new bedrock-agent-runtime client")
            except Exception as e:
                logger.error(f"Error creating bedrock-agent-runtime client: {str(e)}")
                raise
        return self._bedrock_agent_runtime

    def _init_boto3_fallback(self):
        """Initialize boto3 bedrock-agentcore client as fallback with explicit region"""
        try:
            session = boto3.Session()
            self._bedrock_agentcore = session.client(
                "bedrock-agentcore", region_name=self.region
            )
            logger.info(
                f"✅ Initialized boto3 bedrock-agentcore client fallback in region {self.region}"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize boto3 bedrock-agentcore client: {str(e)}"
            )
            self._bedrock_agentcore = None

    def refresh_clients(self):
        """Force refresh of AWS clients to handle expired credentials."""
        logger.info("Refreshing AWS clients due to credential expiration")
        self._bedrock_runtime = None
        self._bedrock_agent_runtime = None
        if self._bedrock_agentcore:
            self._bedrock_agentcore = None
            self._init_boto3_fallback()

    @aws_retry_on_expiration()
    async def _retrieve(self, query: str, max_results: int = 3):
        """
        Retrieve from knowledge base. This method is decorated to handle retries.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": max_results}
                },
            ),
        )

    @aws_retry_on_expiration()
    async def _converse(self, request_body: dict):
        """
        Call Bedrock converse. This method is decorated to handle retries.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.bedrock_runtime.converse(**request_body)
        )

    async def _load_conversation_history(self, k: int = 5) -> str:
        """Load recent conversation history from AgentCore memory"""
        try:
            if not self.memory_id:
                logger.debug("No memory ID available")
                return ""

            import asyncio

            loop = asyncio.get_event_loop()

            # Try AgentCore MemoryClient first
            if self.memory_client:
                try:
                    # Get last k conversation turns using AgentCore MemoryClient
                    recent_turns = await loop.run_in_executor(
                        None,
                        lambda: self.memory_client.get_last_k_turns(
                            memory_id=self.memory_id,
                            actor_id=self.actor_id,
                            session_id=self.session_id,
                            k=k,
                            branch_name=self.branch_name,
                        ),
                    )

                    if recent_turns:
                        # Format conversation history for context
                        context_messages = []
                        for turn in recent_turns:
                            for message in turn:
                                role = message["role"].lower()
                                content = message["content"]["text"]
                                context_messages.append(f"{role.title()}: {content}")

                        context = "\n".join(context_messages)
                        logger.debug(
                            f"✅ Loaded {len(recent_turns)} recent conversation turns via AgentCore"
                        )
                        return context
                    else:
                        logger.debug("No previous conversation history found")
                        return ""

                except Exception as e:
                    logger.warning(
                        f"AgentCore MemoryClient failed, trying boto3 fallback: {str(e)}"
                    )

            # Fall back to boto3 bedrock-agentcore client
            if self._bedrock_agentcore:
                try:
                    # Try to get memory using boto3 client
                    response = await loop.run_in_executor(
                        None,
                        lambda: self._bedrock_agentcore.get_memory(
                            memoryId=self.memory_id
                        ),
                    )

                    # Extract memory content
                    if "memoryContents" in response:
                        contents = []
                        for content in response["memoryContents"]:
                            if "content" in content:
                                contents.append(content["content"])
                        context = (
                            "\n".join(contents[-k:]) if contents else ""
                        )  # Get last k entries
                        logger.debug("✅ Loaded memory content via boto3 fallback")
                        return context

                except Exception as e:
                    logger.error("boto3 fallback also failed: %s", str(e))

            logger.debug("No memory client available")
            return ""

        except Exception as e:
            logger.error(f"Failed to load conversation history: {str(e)}")
            return ""

    async def _store_conversation_turn(self, user_message: str, assistant_message: str):
        """Store conversation turn in AgentCore memory"""
        try:
            if not self.memory_id:
                logger.debug("No memory ID available")
                return

            import asyncio

            loop = asyncio.get_event_loop()

            # Try AgentCore MemoryClient first
            if self.memory_client:
                try:
                    # Store the conversation turn using AgentCore MemoryClient
                    await loop.run_in_executor(
                        None,
                        lambda: self.memory_client.create_event(
                            memory_id=self.memory_id,
                            actor_id=self.actor_id,
                            session_id=self.session_id,
                            messages=[
                                (user_message, "user"),
                                (assistant_message, "assistant"),
                            ],
                        ),
                    )

                    logger.debug(
                        f"✅ Stored conversation turn in AgentCore memory {self.memory_id}"
                    )
                    return

                except Exception as e:
                    logger.warning(
                        f"AgentCore MemoryClient failed, trying boto3 fallback: {str(e)}"
                    )

            # Fall back to boto3 bedrock-agentcore client
            if self._bedrock_agentcore:
                try:
                    # Try to update memory using boto3 client
                    await loop.run_in_executor(
                        None,
                        lambda: self._bedrock_agentcore.update_memory(
                            memoryId=self.memory_id,
                            memoryContents=[
                                {
                                    "content": f"User: {user_message}",
                                    "contentType": "TEXT",
                                },
                                {
                                    "content": f"Assistant: {assistant_message}",
                                    "contentType": "TEXT",
                                },
                            ],
                        ),
                    )

                    logger.debug("✅ Stored conversation turn via boto3 fallback")
                    return

                except Exception as e:
                    logger.error(f"boto3 fallback also failed: {str(e)}")

            logger.debug("No memory client available for storing conversation")

        except Exception as e:
            logger.error(f"Failed to store conversation turn: {str(e)}")

    async def get_memory_content(self) -> Optional[str]:
        """
        Retrieve memory content from AgentCore using MemoryClient.
        """
        return await self._load_conversation_history(k=5)

    async def update_memory(self, user_message: str, assistant_message: str):
        """
        Update AgentCore memory with conversation using MemoryClient.
        """
        await self._store_conversation_turn(user_message, assistant_message)

    async def query_with_rag(self, query: str, max_results: int = 3) -> str:
        """
        Query the agent using RAG with AgentCore memory.
        """
        start_time = time.time()
        
        try:
            if not self.knowledge_base_id:
                logger.warning("No knowledge base configured, using direct query")
                return await self.query(query)

            # Retrieve relevant context using Bedrock Knowledge Base
            logger.info(
                f"Retrieving context for query: '{query}' from KB: {self.knowledge_base_id}"
            )

            retrieve_response = await self._retrieve(query, max_results)

            # Process retrieved data
            contexts = []
            if "retrievalResults" in retrieve_response:
                for result in retrieve_response["retrievalResults"]:
                    if "content" in result and "text" in result["content"]:
                        contexts.append(result["content"]["text"])

            # Get memory content for context
            memory_context = await self.get_memory_content()

            # Build the final prompt with memory context
            prompt_parts = []

            if memory_context:
                prompt_parts.append(
                    f"Previous conversation context:\n{memory_context}\n"
                )

            if not contexts:
                logger.info(f"No relevant information found in KB for '{query}'")
                prompt_parts.append(
                    f'The user asked: "{query}". No information was found in the knowledge base. '
                    "Follow your instructions for how to respond when no relevant information is available."
                )
            else:
                context_str = "\n\n---\n\n".join(contexts)
                prompt_parts.append(
                    f"Use the following knowledge base context to answer the user's query.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f'User Query: "{query}"\n\n'
                    "Remember to follow your rules strictly: if the context is not relevant, you must decline to answer."
                )
                logger.info(
                    f"Generating response using RAG with {len(contexts)} context(s)"
                )

            final_prompt = "\n".join(prompt_parts)

            # Query the model
            response = await self.query_without_memory(final_prompt)

            # Update AgentCore memory with the conversation
            await self.update_memory(query, response)

            # Log successful response
            end_time = time.time()
            execution_time = int((end_time - start_time) * 1000)
            
            if self.agent_logger:
                try:
                    agent_info = AgentInfo(
                        agent_name=self.agent_name,
                        agent_type=self.agent_type,
                        model_id=self.model_id,
                        session_id=self.session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    response_data = ResponseData(
                        query=query,
                        response=response,
                        response_length=len(response),
                        success=True,
                        metadata={
                            "knowledge_base_used": True,
                            "knowledge_base_id": self.knowledge_base_id,
                            "contexts_found": len(contexts),
                            "memory_context_length": len(memory_context) if memory_context else 0,
                            "method": "query_with_rag"
                        }
                    )
                    
                    self.agent_logger.log_agent_response(agent_info, response_data)
                except Exception as e:
                    logger.warning(f"Failed to log agent response: {e}")

            # Apply user-facing response wrapper if enabled
            final_response = response
            if self.response_wrapper:
                try:
                    agent_info = AgentInfo(
                        agent_name=self.agent_name,
                        agent_type=self.agent_type,
                        model_id=self.model_id,
                        session_id=self.session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    additional_info = {
                        "knowledge_base_used": True,
                        "contexts_found": len(contexts)
                    }
                    
                    final_response = self.response_wrapper.wrap_response(
                        response, agent_info, additional_info
                    )
                except Exception as e:
                    logger.warning(f"Failed to wrap response with agent identification: {e}")
                    final_response = response

            return final_response

        except Exception as e:
            # Log error with timing information
            end_time = time.time()
            execution_time = int((end_time - start_time) * 1000)
            
            if self.agent_logger:
                try:
                    error_agent_info = AgentInfo(
                        agent_name=self.agent_name,
                        agent_type=self.agent_type,
                        model_id=self.model_id,
                        session_id=self.session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    self.agent_logger.log_agent_error(
                        error_agent_info,
                        e,
                        context={
                            "query": query[:200],  # Truncated for privacy
                            "method": "query_with_rag",
                            "knowledge_base_id": self.knowledge_base_id,
                            "max_results": max_results
                        }
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log agent error: {log_error}")
            
            logger.error(f"Error in query_with_rag: {str(e)}", exc_info=True)
            return f"Sorry, an error occurred while processing your request: {str(e)}"

    async def query_without_memory(self, prompt: str) -> str:
        """
        Query Bedrock model without built-in memory (AgentCore handles memory separately).
        """
        try:
            # Prepare the request without memory configuration
            request_body = {
                "modelId": self.model_id,
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "system": [{"text": self.system_prompt}],
                "inferenceConfig": {"maxTokens": 2000, "temperature": 0.1},
            }

            # Call Bedrock without memory
            response = await self._converse(request_body)

            # Extract response text
            if "output" in response and "message" in response["output"]:
                content = response["output"]["message"].get("content", [])
                if content and len(content) > 0 and "text" in content[0]:
                    return content[0]["text"]

            return "I apologize, but I couldn't generate a response."

        except Exception as e:
            logger.error(f"Error in query_without_memory: {str(e)}", exc_info=True)
            return f"Sorry, an error occurred: {str(e)}"

    async def query(self, prompt: str) -> str:
        """
        Direct query with AgentCore memory.
        """
        start_time = time.time()
        
        try:
            # Get memory content for context
            memory_context = await self.get_memory_content()

            # Build prompt with memory context
            if memory_context:
                full_prompt = f"Previous conversation context:\n{memory_context}\n\nCurrent query: {prompt}"
            else:
                full_prompt = prompt

            # Query the model
            response = await self.query_without_memory(full_prompt)

            # Update AgentCore memory with the conversation
            await self.update_memory(prompt, response)

            # Log successful response
            end_time = time.time()
            execution_time = int((end_time - start_time) * 1000)
            
            if self.agent_logger:
                try:
                    agent_info = AgentInfo(
                        agent_name=self.agent_name,
                        agent_type=self.agent_type,
                        model_id=self.model_id,
                        session_id=self.session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    response_data = ResponseData(
                        query=prompt,
                        response=response,
                        response_length=len(response),
                        success=True,
                        metadata={
                            "knowledge_base_used": False,
                            "memory_context_length": len(memory_context) if memory_context else 0,
                            "method": "query"
                        }
                    )
                    
                    self.agent_logger.log_agent_response(agent_info, response_data)
                except Exception as e:
                    logger.warning(f"Failed to log agent response: {e}")

            # Apply user-facing response wrapper if enabled
            final_response = response
            if self.response_wrapper:
                try:
                    agent_info = AgentInfo(
                        agent_name=self.agent_name,
                        agent_type=self.agent_type,
                        model_id=self.model_id,
                        session_id=self.session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    additional_info = {
                        "knowledge_base_used": False,
                        "memory_context_available": bool(memory_context)
                    }
                    
                    final_response = self.response_wrapper.wrap_response(
                        response, agent_info, additional_info
                    )
                except Exception as e:
                    logger.warning(f"Failed to wrap response with agent identification: {e}")
                    final_response = response

            return final_response

        except Exception as e:
            # Log error with timing information
            end_time = time.time()
            execution_time = int((end_time - start_time) * 1000)
            
            if self.agent_logger:
                try:
                    error_agent_info = AgentInfo(
                        agent_name=self.agent_name,
                        agent_type=self.agent_type,
                        model_id=self.model_id,
                        session_id=self.session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time
                    )
                    
                    # Safely check memory context availability
                    try:
                        memory_available = bool(await self.get_memory_content())
                    except:
                        memory_available = False
                    
                    self.agent_logger.log_agent_error(
                        error_agent_info,
                        e,
                        context={
                            "query": prompt[:200],  # Truncated for privacy
                            "method": "query",
                            "memory_context_available": memory_available
                        }
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log agent error: {log_error}")
            
            logger.error(f"Error in query: {str(e)}", exc_info=True)
            return f"Sorry, an error occurred: {str(e)}"

    def get_initial_greeting(self) -> str:
        """Get the initial greeting message"""
        return "Hi there, I am your AI agent here to help with questions about the Envision Sustainable Infrastructure Framework."

    def extract_text_from_response(self, response: Any) -> str:
        """
        Extract main text from agent response, preserving existing logic.
        """
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            # Handle various response formats
            content_list = response.get("content")
            if isinstance(content_list, list) and len(content_list) > 0:
                first_content_item = content_list[0]
                if isinstance(first_content_item, dict):
                    text_val = first_content_item.get("text")
                    if isinstance(text_val, str):
                        return text_val.strip()

            # Fallback to other possible text fields
            text_val = response.get("text")
            if isinstance(text_val, str):
                return text_val

            content_val = response.get("content")
            if isinstance(content_val, str):
                return content_val

        # Final fallback
        return str(response) if response is not None else ""

    async def clear_memory(self):
        """
        Clear the AgentCore memory contents by creating a clear event.
        Note: This doesn't delete the memory instance, just indicates memory was cleared.
        """
        try:
            if not self.memory_client or not self.memory_id:
                logger.warning("No AgentCore memory client or memory ID available")
                return

            import asyncio

            loop = asyncio.get_event_loop()

            # Create a clear event in AgentCore memory
            await loop.run_in_executor(
                None,
                lambda: self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[("Memory cleared", "system")],
                ),
            )

            logger.info(f"✅ Cleared AgentCore memory {self.memory_id}")

        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}", exc_info=True)

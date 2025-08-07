"""
Custom Agent implementation using AgentCore Memory
This preserves the existing logic while implementing proper AgentCore memory management.
Based on: https://github.com/awslabs/amazon-bedrock-agentcore-samples/tree/main/01-tutorials/04-AgentCore-memory
"""

import logging
import os
from typing import Any, Optional

import boto3

logger = logging.getLogger(__name__)

# Try to import AgentCore MemoryClient with multiple possible import paths
AGENTCORE_AVAILABLE = False
MemoryClient = None

# Try different possible import paths
import_attempts = [
    ("agentcore.memory", "MemoryClient"),
    ("agentcore", "MemoryClient"),
    ("amazon_bedrock_agentcore.memory", "MemoryClient"),
    ("bedrock_agentcore.memory", "MemoryClient"),
    ("strands_agentcore.memory", "MemoryClient"),
]

for module_path, class_name in import_attempts:
    try:
        module = __import__(module_path, fromlist=[class_name])
        MemoryClient = getattr(module, class_name)
        AGENTCORE_AVAILABLE = True
        logger.info(f"✅ Successfully imported {class_name} from {module_path}")
        break
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import {class_name} from {module_path}: {str(e)}")

if not AGENTCORE_AVAILABLE:
    logger.warning(
        "AgentCore MemoryClient not available, will use boto3 bedrock-agentcore fallback"
    )


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
        """Force refresh of AWS clients to handle expired credentials"""
        logger.info("Refreshing AWS clients due to credential expiration")
        self._bedrock_runtime = None
        self._bedrock_agent_runtime = None
        if self._bedrock_agentcore:
            self._bedrock_agentcore = None
            self._init_boto3_fallback()

    async def _retrieve_with_retry(
        self, query: str, max_results: int = 3, max_retries: int = 2
    ):
        """Retrieve from knowledge base with retry on credential expiration"""
        import asyncio

        from botocore.exceptions import ClientError

        for attempt in range(max_retries + 1):
            try:
                # Use asyncio to run the synchronous boto3 call
                loop = asyncio.get_event_loop()
                retrieve_response = await loop.run_in_executor(
                    None,
                    lambda: self.bedrock_agent_runtime.retrieve(
                        knowledgeBaseId=self.knowledge_base_id,
                        retrievalQuery={"text": query},
                        retrievalConfiguration={
                            "vectorSearchConfiguration": {
                                "numberOfResults": max_results
                            }
                        },
                    ),
                )
                return retrieve_response

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in [
                    "ExpiredTokenException",
                    "InvalidTokenException",
                    "TokenRefreshRequired",
                ]:
                    logger.warning(
                        f"AWS credentials expired (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
                    )
                    if attempt < max_retries:
                        self.refresh_clients()
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        logger.error("Max retries exceeded for credential refresh")
                        raise
                else:
                    # Non-credential error, don't retry
                    logger.error(f"Non-credential error in retrieve: {str(e)}")
                    raise
            except Exception as e:
                logger.error(
                    f"Unexpected error in retrieve (attempt {attempt + 1}): {str(e)}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                else:
                    raise

    async def _converse_with_retry(self, request_body: dict, max_retries: int = 2):
        """Call Bedrock converse with retry on credential expiration"""
        import asyncio

        from botocore.exceptions import ClientError

        for attempt in range(max_retries + 1):
            try:
                # Use asyncio to run the synchronous boto3 call
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.bedrock_runtime.converse(**request_body)
                )
                return response

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in [
                    "ExpiredTokenException",
                    "InvalidTokenException",
                    "TokenRefreshRequired",
                ]:
                    logger.warning(
                        f"AWS credentials expired (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
                    )
                    if attempt < max_retries:
                        self.refresh_clients()
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        logger.error("Max retries exceeded for credential refresh")
                        raise
                else:
                    # Non-credential error, don't retry
                    logger.error(f"Non-credential error in converse: {str(e)}")
                    raise
            except Exception as e:
                logger.error(
                    f"Unexpected error in converse (attempt {attempt + 1}): {str(e)}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                else:
                    raise

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
        try:
            if not self.knowledge_base_id:
                logger.warning("No knowledge base configured, using direct query")
                return await self.query(query)

            # Retrieve relevant context using Bedrock Knowledge Base
            logger.info(
                f"Retrieving context for query: '{query}' from KB: {self.knowledge_base_id}"
            )

            retrieve_response = await self._retrieve_with_retry(query, max_results)

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

            return response

        except Exception as e:
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
            response = await self._converse_with_retry(request_body)

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

            return response

        except Exception as e:
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

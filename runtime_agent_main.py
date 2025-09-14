#!/usr/bin/env python3
"""
Runtime Agent Main - Entry point for AgentCore-hosted agents with observability
This follows the AWS documentation pattern for AgentCore-hosted agents:
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-service-contract.html

Run with: opentelemetry-instrument python runtime_agent_main.py
"""
import asyncio
import logging
import os
import time
from datetime import datetime

import boto3
from aiohttp import web, web_request
from watchtower import CloudWatchLogHandler

from custom_agent import CustomEnvisionAgent
from agent_logging import (
    create_agent_logger, 
    AgentInfo, 
    ResponseData, 
    LoggingConfigManager,
    create_response_wrapper,
    AgentResponseWrapper
)

# Try to import multi-agent orchestrator (after logger is defined)
MULTI_AGENT_AVAILABLE = False
multi_agent_orchestrator_error = None

try:
    from multi_agent_orchestrator import EnvisionMultiAgentOrchestrator

    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    multi_agent_orchestrator_error = str(e)


@web.middleware
async def access_log_middleware(request, handler):
    """Custom access log middleware to filter out /ping calls"""
    start_time = asyncio.get_event_loop().time()
    response = await handler(request)
    process_time = asyncio.get_event_loop().time() - start_time

    # Only log non-ping requests
    if request.path != "/ping":
        logger.info(
            f"{request.method} {request.path} - {response.status} - {process_time:.3f}s"
        )

    return response


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure separate logger for prompts to go to a specific CloudWatch log group
prompt_logger = logging.getLogger("bedrockagent.prompt")
prompt_logger.setLevel(logging.INFO)

# Use watchtower for simplified CloudWatch logging
try:
    # Check if we're running in an environment with AWS credentials
    session = boto3.Session()
    credentials = session.get_credentials()

    if credentials and not prompt_logger.handlers:
        # Use watchtower to handle CloudWatch logging.
        # It will automatically create the log group and stream and handle sequencing tokens.
        # The 'boto3_client' argument is used to pass a pre-configured client.
        logs_client = session.client("logs", region_name=session.region_name)
        cw_handler = CloudWatchLogHandler(
            log_group_name="/bedrockagent/prompt",
            boto3_client=logs_client,
            create_log_group=True,
        )
        cw_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        cw_handler.setFormatter(cw_formatter)
        prompt_logger.addHandler(cw_handler)
        prompt_logger.propagate = False
        logger.info("✅ Watchtower CloudWatch handler configured for prompt logging")
    else:
        raise Exception("No AWS credentials found")

except Exception as e:
    # Fallback to console logging if CloudWatch is not available
    logger.warning(
        f"CloudWatch not available, using console for prompt logging: {str(e)}"
    )
    if not prompt_logger.handlers:
        prompt_handler = logging.StreamHandler()
        prompt_formatter = logging.Formatter(
            "%(asctime)s - PROMPT - %(levelname)s - %(message)s"
        )
        prompt_handler.setFormatter(prompt_formatter)
        prompt_logger.addHandler(prompt_handler)
        prompt_logger.propagate = False

# Import AgentCore Runtime for observability (as per AWS docs)
try:
    from bedrock_agentcore_starter_toolkit import Runtime

    AGENTCORE_RUNTIME_AVAILABLE = True
    logger.info("✅ AgentCore Runtime available")
except ImportError as e:
    AGENTCORE_RUNTIME_AVAILABLE = False
    logger.warning(f"⚠️ AgentCore Runtime not available: {str(e)}")

# Initialize AgentCore Runtime for observability
agentcore_runtime = None
if AGENTCORE_RUNTIME_AVAILABLE:
    try:
        agentcore_runtime = Runtime()
        logger.info("✅ AgentCore Runtime initialized with observability")
    except Exception as e:
        logger.warning(f"Could not initialize AgentCore Runtime: {str(e)}")

# Log multi-agent orchestrator status now that logger is available
if MULTI_AGENT_AVAILABLE:
    logger.info("✅ Multi-agent orchestrator available")
else:
    logger.warning(
        f"⚠️ Multi-agent orchestrator not available: {multi_agent_orchestrator_error}"
    )
    logger.info("Falling back to single agent mode")


class AgentCoreRuntime:
    """
    AgentCore Runtime following AWS documentation pattern for hosted agents
    """

    def __init__(self):
        """Initialize the AgentCore runtime"""
        self.agent = None
        self.multi_agent_orchestrator = None

        # Get configuration from environment
        self.model_id = os.environ.get("MODEL_ID", "us.amazon.nova-micro-v1:0")
        self.region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.knowledge_base_id = os.environ.get("KNOWLEDGE_BASE_ID")
        self.memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
        self.use_multi_agent = (
            os.environ.get("USE_MULTI_AGENT", "true").lower() == "true"
        )

        # Initialize agent response logger
        try:
            logging_config = LoggingConfigManager.load_from_environment()
            self.agent_logger = create_agent_logger(logging_config)
            logger.info("✅ Agent response logger initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize agent response logger: {e}")
            self.agent_logger = None
        
        # Initialize response wrapper for user-facing agent identification
        try:
            self.response_wrapper = create_response_wrapper()
            logger.info("✅ Agent response wrapper initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize agent response wrapper: {e}")
            self.response_wrapper = None

        logger.info(
            f"Runtime configuration: model={self.model_id}, region={self.region}, kb={self.knowledge_base_id}, memory={self.memory_id}, multi_agent={self.use_multi_agent}"
        )

    async def initialize_agent(self) -> CustomEnvisionAgent:
        """Initialize the custom agent and/or multi-agent orchestrator"""
        try:
            # Initialize multi-agent orchestrator if available and enabled
            if MULTI_AGENT_AVAILABLE and self.use_multi_agent:
                try:
                    self.multi_agent_orchestrator = EnvisionMultiAgentOrchestrator(
                        region=self.region
                    )
                    logger.info("✅ Multi-agent orchestrator initialized successfully")
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize multi-agent orchestrator: {e}"
                    )
                    logger.info("Falling back to single agent mode")
                    self.use_multi_agent = False

            # Initialize single agent as fallback or primary
            if not self.use_multi_agent or not self.multi_agent_orchestrator:
                self.agent = CustomEnvisionAgent(
                    model_id=self.model_id,
                    region=self.region,
                    knowledge_base_id=self.knowledge_base_id,
                    memory_id=self.memory_id,
                )
                logger.info("✅ Single agent initialized successfully")

            # --- Add explicit logging for final agent mode ---
            if self.multi_agent_orchestrator and self.use_multi_agent:
                logger.info("🚀 Agent is running in MULTI-AGENT mode.")
            else:
                logger.warning("⚠️ Agent has fallen back to SINGLE-AGENT mode.")

            return self.agent

        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    async def process_query(
        self, query: str, session_id: str = "default", use_rag: bool = True
    ) -> str:
        """Process a query using multi-agent orchestrator or single agent"""
        start_time = time.time()
        agent_name = "unknown"
        agent_type = "unknown"
        success = False
        response = ""
        
        try:
            if not self.agent and not self.multi_agent_orchestrator:
                await self.initialize_agent()

            # Use multi-agent orchestrator if available
            if self.multi_agent_orchestrator and self.use_multi_agent:
                logger.info("Processing query through multi-agent orchestrator")
                agent_name = "multi_agent_orchestrator"
                agent_type = "orchestrator"
                response = await self.multi_agent_orchestrator.process_query(
                    query, session_id
                )
                success = True
                
                # Log session-level orchestrator activity
                if self.agent_logger:
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    agent_info = AgentInfo(
                        agent_name=agent_name,
                        agent_type=agent_type,
                        model_id=self.model_id,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time_ms
                    )
                    response_data = ResponseData(
                        query=query,
                        response=response,
                        response_length=len(response),
                        success=success,
                        metadata={
                            "mode": "multi_agent",
                            "use_rag": use_rag,
                            "has_knowledge_base": bool(self.knowledge_base_id),
                            "runtime_layer": "AgentCoreRuntime"
                        }
                    )
                    self.agent_logger.log_agent_response(agent_info, response_data)
                
                return response

            # Fallback to single agent
            logger.info("Processing query through single agent")
            agent_name = "custom_envision_agent"
            agent_type = "custom"
            
            if use_rag and self.knowledge_base_id:
                response = await self.agent.query_with_rag(query)
            else:
                response = await self.agent.query(query)
            
            success = True
            
            # Log session-level single agent activity
            if self.agent_logger:
                execution_time_ms = int((time.time() - start_time) * 1000)
                agent_info = AgentInfo(
                    agent_name=agent_name,
                    agent_type=agent_type,
                    model_id=self.model_id,
                    session_id=session_id,
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time_ms
                )
                response_data = ResponseData(
                    query=query,
                    response=response,
                    response_length=len(response),
                    success=success,
                    metadata={
                        "mode": "single_agent",
                        "use_rag": use_rag,
                        "has_knowledge_base": bool(self.knowledge_base_id),
                        "runtime_layer": "AgentCoreRuntime"
                    }
                )
                self.agent_logger.log_agent_response(agent_info, response_data)

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            success = False
            error_response = f"Sorry, an error occurred while processing your request: {str(e)}"
            
            # Log runtime processing errors
            if self.agent_logger:
                execution_time_ms = int((time.time() - start_time) * 1000)
                agent_info = AgentInfo(
                    agent_name=agent_name,
                    agent_type=agent_type,
                    model_id=self.model_id,
                    session_id=session_id,
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time_ms
                )
                self.agent_logger.log_agent_error(
                    agent_info, 
                    e, 
                    context={
                        "runtime_layer": "AgentCoreRuntime",
                        "mode": "multi_agent" if self.use_multi_agent else "single_agent",
                        "use_rag": use_rag,
                        "query_length": len(query)
                    }
                )
            
            return error_response

    async def get_session_history(self, session_id: str, k: int = 5) -> list:
        """Get conversation history for a session."""
        try:
            if not self.agent and not self.multi_agent_orchestrator:
                await self.initialize_agent()

            if self.multi_agent_orchestrator and self.use_multi_agent:
                # Use the multi-agent orchestrator's history method
                return await self.multi_agent_orchestrator.get_history(session_id, k)
            elif self.agent:
                # Use the single agent's history method
                # Note: custom_agent.get_memory_content returns a formatted string,
                # so we might need to adjust if a structured list is preferred.
                # For now, we wrap it to match the expected list-of-dicts format.
                history_str = await self.agent.get_memory_content()
                return [{"role": "system", "content": history_str}]
            else:
                logger.warning("No active agent to retrieve history from.")
                return []
        except Exception as e:
            logger.error(f"Error getting session history for {session_id}: {e}")
            return []

    async def health_check(self) -> dict:
        """Perform health check"""
        try:
            health_status = {
                "status": "healthy",
                "agentcore_runtime_available": AGENTCORE_RUNTIME_AVAILABLE,
                "agentcore_runtime_initialized": agentcore_runtime is not None,
                "agent_initialized": self.agent is not None,
                "multi_agent_available": MULTI_AGENT_AVAILABLE,
                "multi_agent_enabled": self.use_multi_agent,
                "multi_agent_initialized": self.multi_agent_orchestrator is not None,
                "configuration": {
                    "model_id": self.model_id,
                    "region": self.region,
                    "has_knowledge_base": bool(self.knowledge_base_id),
                    "has_memory": bool(self.memory_id),
                },
            }

            # Test agent initialization if not already done
            if not self.agent:
                try:
                    await self.initialize_agent()
                    health_status["agent_test"] = "passed"
                except Exception as e:
                    health_status["agent_test"] = f"failed: {str(e)}"
                    health_status["status"] = "degraded"

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    async def run_interactive_session(self):
        """Run an interactive session"""
        try:
            logger.info("🚀 Starting AgentCore Runtime Interactive Session")
            logger.info("=" * 60)

            # Initialize agent
            await self.initialize_agent()

            # Display initial greeting
            greeting = self.agent.get_initial_greeting()
            print(f"\n{greeting}")
            print("\nType 'quit', 'exit', or 'bye' to end the session.")
            print("Type 'health' to check system health.")
            print("Type 'clear' to clear memory.")
            print("-" * 60)

            interaction_count = 0

            while True:
                try:
                    # Get user input
                    user_input = input("\n🤔 You: ").strip()

                    if not user_input:
                        continue

                    # Handle special commands
                    if user_input.lower() in ["quit", "exit", "bye"]:
                        print("\n👋 Goodbye!")
                        break

                    elif user_input.lower() == "health":
                        health = await self.health_check()
                        print(f"\n🏥 Health Status: {health}")
                        continue

                    elif user_input.lower() == "clear":
                        await self.agent.clear_memory()
                        print("\n🧹 Memory cleared!")
                        continue

                    # Process query
                    interaction_count += 1
                    session_id = f"interactive_session_{interaction_count}"
                    print("\n🤖 Agent: ", end="", flush=True)
                    response = await self.process_query(user_input, session_id)
                    print(response)

                except KeyboardInterrupt:
                    print("\n\n👋 Session interrupted. Goodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in interactive session: {str(e)}")
                    print(f"\n❌ Error: {str(e)}")

            logger.info(f"Session completed with {interaction_count} interactions")

        except Exception as e:
            logger.error(f"Interactive session failed: {str(e)}")
            raise


# Global runtime instance for HTTP endpoints
runtime_instance = None


async def health_endpoint(request: web_request.Request) -> web.Response:
    """
    Health check endpoint required by AgentCore service contract.
    GET /health - Must return 200 when healthy.
    """
    start_time = time.time()
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }
    request_metadata = {
        "method": request.method,
        "path": request.path,
        "remote_addr": request.remote,
        "user_agent": request.headers.get("User-Agent", "unknown")
    }
    
    try:
        if not runtime_instance:
            # Log health check failure due to uninitialized runtime
            execution_time_ms = int((time.time() - start_time) * 1000)
            fallback_logger = logging.getLogger("health_endpoint_error")
            fallback_logger.error(f"Health check failed - runtime not initialized (took {execution_time_ms}ms)")
            
            return web.json_response(
                {"status": "unhealthy", "error": "Runtime not initialized"},
                status=503,
                headers=cors_headers,
            )

        health_status = await runtime_instance.health_check()
        status_code = 200 if health_status.get("status") == "healthy" else 503
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Log health check result
        if runtime_instance.agent_logger:
            agent_info = AgentInfo(
                agent_name="http_endpoint",
                agent_type="endpoint",
                model_id=runtime_instance.model_id,
                session_id="health_check",
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms
            )
            
            if health_status.get("status") == "healthy":
                response_data = ResponseData(
                    query="health_check",
                    response=f"Health status: {health_status.get('status')}",
                    response_length=len(str(health_status)),
                    success=True,
                    metadata={
                        "endpoint": "/health",
                        "request_metadata": request_metadata,
                        "health_status": health_status,
                        "status_code": status_code
                    }
                )
                runtime_instance.agent_logger.log_agent_response(agent_info, response_data)
            else:
                runtime_instance.agent_logger.log_agent_error(
                    agent_info,
                    Exception(f"Health check failed: {health_status}"),
                    context={
                        "endpoint": "/health",
                        "request_metadata": request_metadata,
                        "health_status": health_status,
                        "status_code": status_code,
                        "error_type": "health_check_degraded"
                    }
                )

        return web.json_response(health_status, status=status_code, headers=cors_headers)

    except Exception as e:
        logger.error(f"Health endpoint error: {e}")
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Log health endpoint error
        if runtime_instance and runtime_instance.agent_logger:
            agent_info = AgentInfo(
                agent_name="http_endpoint",
                agent_type="endpoint",
                model_id=runtime_instance.model_id if runtime_instance else "n/a",
                session_id="health_check",
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms
            )
            runtime_instance.agent_logger.log_agent_error(
                agent_info,
                e,
                context={
                    "endpoint": "/health",
                    "request_metadata": request_metadata,
                    "error_type": "health_endpoint_error"
                }
            )
        
        return web.json_response(
            {"status": "unhealthy", "error": str(e)}, status=503, headers=cors_headers
        )


async def ping_endpoint(request: web_request.Request) -> web.Response:
    """
    Ping endpoint required by AgentCore service contract
    GET /ping - Simple liveness check (minimal logging to avoid spam)
    """
    start_time = time.time()
    
    try:
        # Only log ping requests if debug level logging is enabled and agent logger is available
        if (runtime_instance and runtime_instance.agent_logger and 
            runtime_instance.agent_logger.config.log_level == "DEBUG"):
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            request_metadata = {
                "method": request.method,
                "path": request.path,
                "remote_addr": request.remote
            }
            
            agent_info = AgentInfo(
                agent_name="http_endpoint",
                agent_type="endpoint",
                model_id="n/a",
                session_id="ping",
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms
            )
            response_data = ResponseData(
                query="ping",
                response="pong",
                response_length=4,
                success=True,
                metadata={
                    "endpoint": "/ping",
                    "request_metadata": request_metadata,
                    "liveness_check": True
                }
            )
            runtime_instance.agent_logger.log_agent_response(agent_info, response_data)
        
        return web.json_response({"message": "pong"}, status=200)
        
    except Exception as e:
        # Even for ping, log errors if they occur
        if runtime_instance and runtime_instance.agent_logger:
            execution_time_ms = int((time.time() - start_time) * 1000)
            agent_info = AgentInfo(
                agent_name="http_endpoint",
                agent_type="endpoint",
                model_id="n/a",
                session_id="ping",
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms
            )
            runtime_instance.agent_logger.log_agent_error(
                agent_info,
                e,
                context={
                    "endpoint": "/ping",
                    "error_type": "ping_endpoint_error"
                }
            )
        
        # Return error response
        return web.json_response({"message": "error", "error": str(e)}, status=500)


async def invocations_endpoint(request: web_request.Request) -> web.Response:
    """
    Invocations endpoint required by AgentCore service contract.
    Handles both prompt processing and other actions like history retrieval.
    """
    start_time = time.time()
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }
    request_metadata = {
        "method": request.method,
        "path": request.path,
        "remote_addr": request.remote,
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "content_type": request.headers.get("Content-Type", "unknown"),
        "content_length": request.headers.get("Content-Length", "0")
    }
    
    try:
        if not runtime_instance:
            # Log endpoint error with timing
            if runtime_instance and runtime_instance.agent_logger:
                execution_time_ms = int((time.time() - start_time) * 1000)
                agent_info = AgentInfo(
                    agent_name="http_endpoint",
                    agent_type="endpoint",
                    model_id="n/a",
                    session_id="unknown",
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time_ms
                )
                runtime_instance.agent_logger.log_agent_error(
                    agent_info,
                    Exception("Runtime not initialized"),
                    context={
                        "endpoint": "/invocations",
                        "request_metadata": request_metadata,
                        "error_type": "initialization_error"
                    }
                )
            return web.json_response({"error": "Runtime not initialized"}, status=503, headers=cors_headers)

        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            # Log parsing error with timing
            if runtime_instance.agent_logger:
                execution_time_ms = int((time.time() - start_time) * 1000)
                agent_info = AgentInfo(
                    agent_name="http_endpoint",
                    agent_type="endpoint",
                    model_id="n/a",
                    session_id="unknown",
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time_ms
                )
                runtime_instance.agent_logger.log_agent_error(
                    agent_info,
                    e,
                    context={
                        "endpoint": "/invocations",
                        "request_metadata": request_metadata,
                        "error_type": "json_parsing_error"
                    }
                )
            return web.json_response(
                {"error": "Invalid JSON in request body"}, status=400, headers=cors_headers
            )

        session_id = body.get("sessionId", "default")
        request_metadata["session_id"] = session_id

        # Handle different actions based on request body
        if body.get("action") == "getHistory":
            logger.info(f"Handling getHistory action for session: {session_id}")
            try:
                history = await runtime_instance.get_session_history(
                    session_id, k=body.get("k", 5)
                )
                
                # Log successful history retrieval
                if runtime_instance.agent_logger:
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    agent_info = AgentInfo(
                        agent_name="http_endpoint",
                        agent_type="endpoint",
                        model_id="n/a",
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time_ms
                    )
                    response_data = ResponseData(
                        query=f"getHistory(k={body.get('k', 5)})",
                        response=f"Retrieved {len(history)} history items",
                        response_length=len(str(history)),
                        success=True,
                        metadata={
                            "endpoint": "/invocations",
                            "action": "getHistory",
                            "request_metadata": request_metadata,
                            "history_count": len(history)
                        }
                    )
                    runtime_instance.agent_logger.log_agent_response(agent_info, response_data)
                
                return web.json_response({"history": history}, status=200, headers=cors_headers)
                
            except Exception as e:
                logger.error(f"Error retrieving history: {e}")
                # Log history retrieval error
                if runtime_instance.agent_logger:
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    agent_info = AgentInfo(
                        agent_name="http_endpoint",
                        agent_type="endpoint",
                        model_id="n/a",
                        session_id=session_id,
                        timestamp=datetime.now(),
                        execution_time_ms=execution_time_ms
                    )
                    runtime_instance.agent_logger.log_agent_error(
                        agent_info,
                        e,
                        context={
                            "endpoint": "/invocations",
                            "action": "getHistory",
                            "request_metadata": request_metadata,
                            "error_type": "history_retrieval_error"
                        }
                    )
                return web.json_response({"error": f"Failed to retrieve history: {e}"}, status=500, headers=cors_headers)

        # Default action is to process a prompt
        prompt = None
        for field in ["prompt", "query", "message", "input", "text"]:
            if field in body:
                prompt = body[field]
                break

        if not prompt:
            logger.error(f"No prompt or valid action found in request body: {body}")
            # Log missing prompt error
            if runtime_instance.agent_logger:
                execution_time_ms = int((time.time() - start_time) * 1000)
                agent_info = AgentInfo(
                    agent_name="http_endpoint",
                    agent_type="endpoint",
                    model_id="n/a",
                    session_id=session_id,
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time_ms
                )
                runtime_instance.agent_logger.log_agent_error(
                    agent_info,
                    Exception("No prompt or valid action found in request"),
                    context={
                        "endpoint": "/invocations",
                        "request_metadata": request_metadata,
                        "request_body_keys": list(body.keys()),
                        "error_type": "missing_prompt_error"
                    }
                )
            return web.json_response(
                {"error": "No prompt or valid action found in request"}, status=400, headers=cors_headers
            )

        # Log user query to separate prompt log group
        prompt_logger.info(f"USER_QUERY: {prompt}")

        # Process the query
        try:
            response = await runtime_instance.process_query(prompt, session_id)
            # Log agent response to separate prompt log group
            prompt_logger.info(f"AGENT_RESPONSE: {response}")
            # Keep abbreviated response in main log for debugging
            logger.info(
                f"Query processed successfully, response length: {len(response)} chars"
            )
            
            # Log successful HTTP endpoint processing
            if runtime_instance.agent_logger:
                execution_time_ms = int((time.time() - start_time) * 1000)
                agent_info = AgentInfo(
                    agent_name="http_endpoint",
                    agent_type="endpoint",
                    model_id=runtime_instance.model_id,
                    session_id=session_id,
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time_ms
                )
                response_data = ResponseData(
                    query=prompt,
                    response=response,
                    response_length=len(response),
                    success=True,
                    metadata={
                        "endpoint": "/invocations",
                        "action": "process_query",
                        "request_metadata": request_metadata,
                        "agent_mode": "multi_agent" if runtime_instance.use_multi_agent else "single_agent",
                        "has_knowledge_base": bool(runtime_instance.knowledge_base_id)
                    }
                )
                runtime_instance.agent_logger.log_agent_response(agent_info, response_data)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            prompt_logger.error(f"QUERY_ERROR: {e}")
            response = f"Sorry, an error occurred while processing your request: {e}"
            
            # Log query processing error
            if runtime_instance.agent_logger:
                execution_time_ms = int((time.time() - start_time) * 1000)
                agent_info = AgentInfo(
                    agent_name="http_endpoint",
                    agent_type="endpoint",
                    model_id=runtime_instance.model_id,
                    session_id=session_id,
                    timestamp=datetime.now(),
                    execution_time_ms=execution_time_ms
                )
                runtime_instance.agent_logger.log_agent_error(
                    agent_info,
                    e,
                    context={
                        "endpoint": "/invocations",
                        "action": "process_query",
                        "request_metadata": request_metadata,
                        "query_length": len(prompt),
                        "error_type": "query_processing_error"
                    }
                )

        # Return only the response as plain text (no sessionId or timestamp)
        return web.Response(text=response, status=200, content_type="text/plain", headers=cors_headers)

    except Exception as e:
        logger.error(f"Invocations endpoint error: {e}")
        # Log general endpoint error
        if runtime_instance and runtime_instance.agent_logger:
            execution_time_ms = int((time.time() - start_time) * 1000)
            agent_info = AgentInfo(
                agent_name="http_endpoint",
                agent_type="endpoint",
                model_id="n/a",
                session_id="unknown",
                timestamp=datetime.now(),
                execution_time_ms=execution_time_ms
            )
            runtime_instance.agent_logger.log_agent_error(
                agent_info,
                e,
                context={
                    "endpoint": "/invocations",
                    "request_metadata": request_metadata,
                    "error_type": "general_endpoint_error"
                }
            )
        return web.json_response(
            {"error": f"Internal server error: {e}"}, status=500, headers=cors_headers
        )


async def start_http_server():
    """Start the HTTP server required by AgentCore service contract"""
    global runtime_instance

    # Initialize runtime
    runtime_instance = AgentCoreRuntime()

    # Create aiohttp application with custom logging
    app = web.Application()

    # Add required endpoints
    app.router.add_get("/health", health_endpoint)
    app.router.add_get("/ping", ping_endpoint)
    app.router.add_post("/invocations", invocations_endpoint)

    # Add custom access log middleware to exclude /ping calls
    app.middlewares.append(access_log_middleware)

    # Start server on port 8080 (required by AgentCore) with custom access log
    runner = web.AppRunner(app, access_log=None)  # Disable default access log
    await runner.setup()

    # Bind to localhost by default for security, allow override via environment
    host = os.environ.get("HOST", "127.0.0.1")
    site = web.TCPSite(runner, host, 8080)
    await site.start()

    logger.info("🚀 AgentCore Runtime HTTP Server started on port 8080")
    logger.info("📋 Available endpoints:")
    logger.info("   GET  /health      - Health check")
    logger.info("   GET  /ping        - Liveness check")
    logger.info("   POST /invocations - Agent invocations")

    return runner


async def main():
    """Main entry point for the AgentCore runtime"""
    try:
        # Check if we're running in CLI mode
        if len(os.sys.argv) > 1:
            # Create runtime instance for CLI operations
            runtime = AgentCoreRuntime()

            if os.sys.argv[1] == "health":
                health = await runtime.health_check()
                print(f"Health Status: {health}")
                return

            elif os.sys.argv[1] == "query":
                if len(os.sys.argv) < 3:
                    print(
                        "Usage: python runtime_agent_main.py query 'your question here'"
                    )
                    return

                query = " ".join(os.sys.argv[2:])
                response = await runtime.process_query(query)
                print(f"Response: {response}")
                return

            elif os.sys.argv[1] == "interactive":
                await runtime.run_interactive_session()
                return

        # Default: Start HTTP server for AgentCore service contract
        runner = await start_http_server()

        try:
            # Keep the server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down HTTP server...")
        finally:
            await runner.cleanup()

    except Exception as e:
        logger.error(f"Runtime failed: {str(e)}")
        raise


if __name__ == "__main__":
    # This script follows AWS AgentCore documentation for hosted agents
    # Run with: opentelemetry-instrument python runtime_agent_main.py

    print("🔍 AgentCore Runtime for Hosted Agents")
    print("=" * 40)

    if AGENTCORE_RUNTIME_AVAILABLE:
        print("✅ AgentCore Runtime available")
    else:
        print(
            "⚠️ Running without AgentCore Runtime (install bedrock_agentcore_starter_toolkit)"
        )

    print("💡 For full observability, run with:")
    print("   opentelemetry-instrument python runtime_agent_main.py")
    print("📖 Following AWS documentation:")
    print(
        "   https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html"
    )
    print()

    # Run the main function
    asyncio.run(main())

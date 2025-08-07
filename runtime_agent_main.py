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
from datetime import datetime

import boto3
from aiohttp import web, web_request

from custom_agent import CustomEnvisionAgent

# Try to import multi-agent orchestrator
try:
    from multi_agent_orchestrator import EnvisionMultiAgentOrchestrator
    MULTI_AGENT_AVAILABLE = True
    logger.info("✅ Multi-agent orchestrator available")
except ImportError as e:
    MULTI_AGENT_AVAILABLE = False
    logger.warning(f"⚠️ Multi-agent orchestrator not available: {str(e)}")
    logger.info("Falling back to single agent mode")


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

# Configure separate logger for prompts (will go to /bedrockagent/prompt log group)
prompt_logger = logging.getLogger("bedrockagent.prompt")
prompt_logger.setLevel(logging.INFO)


class CloudWatchHandler(logging.Handler):
    """Custom CloudWatch handler for prompt logging"""

    def __init__(self, log_group_name="/bedrockagent/prompt", region="us-east-1"):
        super().__init__()
        self.log_group_name = log_group_name
        self.region = region
        self.log_stream_name = (
            f"prompt-stream-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
        self.cloudwatch_client = None
        self.sequence_token = None
        self._initialize_cloudwatch()

    def _initialize_cloudwatch(self):
        """Initialize CloudWatch client and create log group/stream if needed"""
        try:
            # Get region from environment or use default
            region = os.environ.get("AWS_DEFAULT_REGION", self.region)
            session = boto3.Session()
            self.cloudwatch_client = session.client("logs", region_name=region)

            # Create log group if it doesn't exist
            try:
                self.cloudwatch_client.create_log_group(
                    logGroupName=self.log_group_name
                )
                logger.info(f"✅ Created CloudWatch log group: {self.log_group_name}")
            except self.cloudwatch_client.exceptions.ResourceAlreadyExistsException:
                logger.debug(
                    f"CloudWatch log group already exists: {self.log_group_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not create log group {self.log_group_name}: {str(e)}"
                )

            # Create log stream
            try:
                self.cloudwatch_client.create_log_stream(
                    logGroupName=self.log_group_name, logStreamName=self.log_stream_name
                )
                logger.info(f"✅ Created CloudWatch log stream: {self.log_stream_name}")
            except self.cloudwatch_client.exceptions.ResourceAlreadyExistsException:
                logger.debug(
                    f"CloudWatch log stream already exists: {self.log_stream_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not create log stream {self.log_stream_name}: {str(e)}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize CloudWatch logging: {str(e)}")
            self.cloudwatch_client = None

    def emit(self, record):
        """Send log record to CloudWatch"""
        if not self.cloudwatch_client:
            return

        try:
            # Format the log message
            message = self.format(record)
            timestamp = int(record.created * 1000)  # CloudWatch expects milliseconds

            # Prepare log event
            log_event = {"timestamp": timestamp, "message": message}

            # Send to CloudWatch
            kwargs = {
                "logGroupName": self.log_group_name,
                "logStreamName": self.log_stream_name,
                "logEvents": [log_event],
            }

            if self.sequence_token:
                kwargs["sequenceToken"] = self.sequence_token

            response = self.cloudwatch_client.put_log_events(**kwargs)
            self.sequence_token = response.get("nextSequenceToken")

        except self.cloudwatch_client.exceptions.InvalidSequenceTokenException:
            # Handle sequence token issues by getting the correct token
            try:
                streams = self.cloudwatch_client.describe_log_streams(
                    logGroupName=self.log_group_name,
                    logStreamNamePrefix=self.log_stream_name,
                )
                if streams["logStreams"]:
                    self.sequence_token = streams["logStreams"][0].get(
                        "uploadSequenceToken"
                    )
                    # Retry with correct token
                    kwargs["sequenceToken"] = self.sequence_token
                    response = self.cloudwatch_client.put_log_events(**kwargs)
                    self.sequence_token = response.get("nextSequenceToken")
            except Exception as retry_e:
                logger.error(
                    f"Failed to retry CloudWatch log after sequence token error: {str(retry_e)}"
                )

        except Exception as e:
            logger.error(f"Failed to send log to CloudWatch: {str(e)}")


# Create CloudWatch handler for prompt logging if in AWS environment
try:
    # Check if we're running in AWS (has AWS credentials)
    session = boto3.Session()
    credentials = session.get_credentials()

    if credentials and not prompt_logger.handlers:
        # Use CloudWatch handler in AWS environment
        cloudwatch_handler = CloudWatchHandler()
        cloudwatch_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        cloudwatch_handler.setFormatter(cloudwatch_formatter)
        prompt_logger.addHandler(cloudwatch_handler)
        prompt_logger.propagate = False
        logger.info("✅ CloudWatch handler configured for prompt logging")
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
        self.use_multi_agent = os.environ.get("USE_MULTI_AGENT", "true").lower() == "true"

        logger.info(
            f"Runtime configuration: model={self.model_id}, region={self.region}, kb={self.knowledge_base_id}, memory={self.memory_id}, multi_agent={self.use_multi_agent}"
        )

    async def initialize_agent(self) -> CustomEnvisionAgent:
        """Initialize the custom agent and/or multi-agent orchestrator"""
        try:
            # Initialize multi-agent orchestrator if available and enabled
            if MULTI_AGENT_AVAILABLE and self.use_multi_agent:
                try:
                    self.multi_agent_orchestrator = EnvisionMultiAgentOrchestrator(region=self.region)
                    logger.info("✅ Multi-agent orchestrator initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize multi-agent orchestrator: {e}")
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

            return self.agent

        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    async def process_query(self, query: str, session_id: str = "default", use_rag: bool = True) -> str:
        """Process a query using multi-agent orchestrator or single agent"""
        try:
            if not self.agent and not self.multi_agent_orchestrator:
                await self.initialize_agent()

            # Use multi-agent orchestrator if available
            if self.multi_agent_orchestrator and self.use_multi_agent:
                logger.info("Processing query through multi-agent orchestrator")
                response = await self.multi_agent_orchestrator.process_query(query, session_id)
                return response

            # Fallback to single agent
            logger.info("Processing query through single agent")
            if use_rag and self.knowledge_base_id:
                response = await self.agent.query_with_rag(query)
            else:
                response = await self.agent.query(query)

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Sorry, an error occurred while processing your request: {str(e)}"

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
    Health check endpoint required by AgentCore service contract
    GET /health - Must return 200 when healthy
    """
    try:
        if not runtime_instance:
            return web.json_response(
                {"status": "unhealthy", "error": "Runtime not initialized"}, status=503
            )

        health_status = await runtime_instance.health_check()
        status_code = 200 if health_status.get("status") == "healthy" else 503

        return web.json_response(health_status, status=status_code)

    except Exception as e:
        logger.error(f"Health endpoint error: {str(e)}")
        return web.json_response({"status": "unhealthy", "error": str(e)}, status=503)


async def ping_endpoint(request: web_request.Request) -> web.Response:
    """
    Ping endpoint required by AgentCore service contract
    GET /ping - Simple liveness check (no logging to avoid spam)
    """
    return web.json_response({"message": "pong"}, status=200)


async def invocations_endpoint(request: web_request.Request) -> web.Response:
    """
    Invocations endpoint required by AgentCore service contract
    POST /invocations - Main endpoint for agent requests
    """
    try:
        if not runtime_instance:
            return web.json_response({"error": "Runtime not initialized"}, status=503)

        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse request body: {str(e)}")
            return web.json_response(
                {"error": "Invalid JSON in request body"}, status=400
            )

        # Extract prompt and session_id from request
        prompt = None
        session_id = body.get("sessionId", "default")

        for field in ["prompt", "query", "message", "input", "text"]:
            if field in body:
                prompt = body[field]
                break

        if not prompt:
            logger.error(f"No prompt found in request body: {body}")
            return web.json_response(
                {"error": "No prompt/query found in request"}, status=400
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
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            prompt_logger.error(f"QUERY_ERROR: {str(e)}")
            response = (
                f"Sorry, an error occurred while processing your request: {str(e)}"
            )

        # Return only the response as plain text (no sessionId or timestamp)
        return web.Response(text=response, status=200, content_type="text/plain")

    except Exception as e:
        logger.error(f"Invocations endpoint error: {str(e)}")
        return web.json_response(
            {"error": f"Internal server error: {str(e)}"}, status=500
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

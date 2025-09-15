"""
AWS Lambda function to proxy requests to AgentCore service.
This keeps AWS credentials server-side and provides a clean API for the frontend.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from typing import Any, Dict

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration - get from environment variables
AGENT_ARN = os.environ.get(
    "AGENT_ARN",
    "arn:aws:bedrock-agentcore:us-east-1:886436945166:runtime/hosted_agent_sample-KEQNVq8Whv",
)
AGENTCORE_MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "memory_io2n5-94iksj6Jr7")

# Initialize the Bedrock AgentCore client for all operations
agentcore_client = boto3.client("bedrock-agentcore")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler that calls the async handler."""
    return asyncio.run(async_lambda_handler(event, context))


async def async_lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for AgentCore proxy requests.
    """
    try:
        # Handle CORS preflight requests
        if event.get("httpMethod") == "OPTIONS":
            return {
                "statusCode": 200,
                # SECURITY: Allow all origins for broad compatibility, but in a production
                # environment, this should be locked down to the specific frontend domain
                # in the API Gateway configuration.
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    "Access-Control-Max-Age": "86400",
                },
                "body": "",
            }

        # Extract user info from API Gateway context (already verified by Cognito Authorizer)
        user_info = extract_user_from_context(event)

        # Parse the request body
        if "body" not in event:
            return create_error_response(400, "Missing request body")

        try:
            body = (
                json.loads(base64.b64decode(event["body"]).decode("utf-8"))
                if event.get("isBase64Encoded", False)
                else json.loads(event["body"])
            )
        except json.JSONDecodeError as e:
            return create_error_response(400, f"Invalid JSON in request body: {e}")

        # Get session ID and action from body
        session_id = body.get("sessionId", str(uuid.uuid4()))
        action = body.get("action")

        # Route based on action
        if action == "getHistory":
            logger.info("Handling getHistory action for session: %s", session_id)
            history_messages = await get_conversation_history(
                session_id, k=body.get("k", 10)
            )
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"messages": history_messages}),
            }

        # Default action: process a prompt
        prompt = body.get("prompt", "")
        if not prompt:
            return create_error_response(
                400, "Missing 'prompt' in request body for invoke action"
            )

        logger.info(
            "Processing prompt for user: %s, session: %s",
            user_info.get("email", "unknown"),
            session_id,
        )

        # Call the AgentCore runtime
        agent_response = await call_agentcore_runtime(prompt, session_id)

        # Store the conversation turn in memory
        await store_conversation_turn(session_id, prompt, agent_response)

        # Return successful response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "response": agent_response,
                    "sessionId": session_id,
                }
            ),
        }

    except Exception as e:
        logger.error("FATAL: Unhandled exception in async_lambda_handler: %s", e, exc_info=True)
        return create_error_response(500, f"Internal server error: {e}")


def extract_user_from_context(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract user information from API Gateway request context.
    (API Gateway Cognito Authorizer populates this).
    """
    try:
        # API Gateway puts Cognito user info in requestContext.authorizer.claims
        request_context = event.get("requestContext", {})
        authorizer = request_context.get("authorizer", {})
        claims = authorizer.get("claims", {})
        if claims:
            logger.info(
                "User authenticated: %s",
                claims.get("email", claims.get("sub", "unknown")),
            )
            return claims

        # Fallback: try to get from other locations in the context
        cognito_identity_id = request_context.get("identity", {}).get(
            "cognitoIdentityId"
        )
        if cognito_identity_id:
            return {"sub": cognito_identity_id}

        logger.warning("No user information found in request context")
        return {"sub": "unknown"}

    except Exception as e:
        logger.error("Error extracting user from context: %s", e)
        return {"sub": "unknown"}


def call_agentcore_runtime_sync(prompt: str, session_id: str) -> str:
    """Synchronous call to the AgentCore runtime via ARN."""
    try:
        # Prepare the payload for AgentCore
        payload = json.dumps({"prompt": prompt, "sessionId": session_id}).encode(
            "utf-8"
        )
        logger.info("Payload prepared: %d bytes", len(payload))

        # Generate a random trace ID
        trace_id = str(uuid.uuid4())[:8]

        # Invoke the AgentCore service
        logger.info("Invoking AgentCore service with traceId: %s", trace_id)
        response = agentcore_client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_ARN,
            traceId=trace_id,
            payload=payload,
        )
        logger.info("AgentCore response received.")

        # Process the response
        return process_agentcore_response(response)

    except Exception as e:
        logger.error("Error calling AgentCore runtime: %s", e, exc_info=True)
        logger.info("AgentCore runtime failed. Returning graceful fallback response.")
        return get_fallback_response(prompt)


async def call_agentcore_runtime(prompt: str, session_id: str) -> str:
    """
    Async wrapper for the synchronous runtime call.
    This uses a ThreadPoolExecutor because the boto3 SDK is synchronous.
    Running it in an executor prevents it from blocking the main asyncio event loop,
    which is important for performance in an async Lambda handler.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(
            executor, call_agentcore_runtime_sync, prompt, session_id
        )


def get_fallback_response(prompt: str) -> str:
    """Provide a fallback response when the containerized runtime is not available."""
    # Simple keyword-based routing for fallback
    prompt_lower = prompt.lower()
    if any(
        keyword in prompt_lower
        for keyword in ["envision", "credit", "category", "scoring", "assessment"]
    ):
        return (
            f"Thank you for your question about the Envision Sustainable Infrastructure Framework.\n\n"
            f'Your question: "{prompt}"\n\n'
            "I'm currently operating in simplified mode. For detailed information, please try again later."
        )

    return (
        f'Thank you for your sustainability question.\n\nYour question: "{prompt}"\n\n'
        "I'm currently operating in simplified mode. For detailed guidance, please try again later."
    )




def process_agentcore_response(response: Dict[str, Any]) -> str:
    """
    Process the AgentCore response and extract the text content from StreamingBody.
    """
    try:
        content_type = response.get("contentType", "")
        logger.info("Processing response with content type: %s", content_type)
        streaming_body = response.get("response")

        if not streaming_body:
            return ""

        if "text/event-stream" in content_type:
            content = []
            if hasattr(streaming_body, "iter_lines"):
                for line in streaming_body.iter_lines(chunk_size=1024):
                    if line:
                        decoded_line = line.decode("utf-8")
                        if decoded_line.startswith("data:"):
                            try:
                                # Parse the JSON data part of the event
                                data_json = json.loads(decoded_line[5:].strip())
                                # Extract the bytes from the completion chunk
                                if "completion" in data_json and "bytes" in data_json["completion"]:
                                    content.append(data_json["completion"]["bytes"])
                            except json.JSONDecodeError:
                                # Handle cases where the data is not valid JSON
                                logger.warning("Could not decode JSON from event stream line: %s", decoded_line)
                return "".join(content)

        if hasattr(streaming_body, "read"):
            return streaming_body.read().decode("utf-8")

        return str(streaming_body)

    except Exception as e:
        logger.error("Error processing AgentCore response: %s", e, exc_info=True)
        return f"Error processing response: {e}"


async def get_conversation_history(session_id: str, k: int = 10) -> list:
    """
    Get conversation history from AgentCore memory by listing events for a session.
    """
    if not AGENTCORE_MEMORY_ID:
        logger.warning("AGENTCORE_MEMORY_ID is not set. Cannot retrieve history.")
        return []

    try:
        logger.info(
            "Getting history for memoryId: %s, sessionId: %s",
            AGENTCORE_MEMORY_ID,
            session_id,
        )

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: agentcore_client.list_events(
                    memoryId=AGENTCORE_MEMORY_ID,
                    filter={"sessionId": session_id},
                    maxResults=k * 2, # Get k turns
                ),
            )

        messages = []
        if "events" in response:
            for event in sorted(response["events"], key=lambda x: x["eventTimestamp"]):
                for part in event.get("payload", []):
                    if "conversational" in part:
                        conv = part["conversational"]
                        messages.append({
                            "role": conv["role"].lower(),
                            "content": conv["content"]["text"],
                        })

        logger.info("Retrieved %d messages from history", len(messages))
        return messages

    except Exception as e:
        logger.error("Error getting conversation history: %s", e, exc_info=True)
        return []


async def store_conversation_turn(session_id: str, user_message: str, assistant_message: str):
    """
    Store a conversation turn by creating events in AgentCore memory.
    """
    if not AGENTCORE_MEMORY_ID:
        logger.warning("AGENTCORE_MEMORY_ID is not set. Cannot store history.")
        return

    try:
        logger.info("Storing turn for memoryId: %s, sessionId: %s", AGENTCORE_MEMORY_ID, session_id)
        loop = asyncio.get_event_loop()

        # The payload is a list of two events, one for the user and one for the assistant
        payload = [
            {
                "conversational": {
                    "content": {"text": user_message},
                    "role": "USER",
                }
            },
            {
                "conversational": {
                    "content": {"text": assistant_message},
                    "role": "ASSISTANT",
                }
            },
        ]

        # We can use a single create_event call to store the turn
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                lambda: agentcore_client.create_event(
                    memoryId=AGENTCORE_MEMORY_ID,
                    sessionId=session_id,
                    actorId="user1", # A placeholder actorId
                    eventTimestamp=datetime.now(UTC),
                    payload=payload
                ),
            )
        logger.info("Successfully created conversation event.")

    except Exception as e:
        logger.error("Error creating conversation event: %s", e, exc_info=True)


def create_error_response(status_code: int, message: str) -> Dict[str, Any]:
    """
    Create a standardized error response.
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
        "body": json.dumps({"error": message, "statusCode": status_code}),
    }

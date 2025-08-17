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

# Initialize the Bedrock AgentCore client
agent_core_client = boto3.client("bedrock-agentcore")


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

        # Check if this is a history request
        action = body.get("action", "query")
        session_id = body.get("sessionId", "")

        if action == "getHistory":
            # Handle history request
            logger.info(
                "Processing history request - User: %s, SessionId: %s",
                user_info.get("email", user_info.get("sub", "unknown")),
                session_id,
            )
            
            k = body.get("k", 3)  # Number of messages to retrieve
            history_response = await call_agentcore_history(session_id, k)
            
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                },
                "body": json.dumps(history_response),
            }
        else:
            # Handle regular query request
            prompt = body.get("prompt", "")
            
            if not prompt:
                return create_error_response(400, "Missing 'prompt' in request body")

            logger.info(
                "Processing request - User: %s, Prompt: %s..., SessionId: %s",
                user_info.get("email", user_info.get("sub", "unknown")),
                prompt[:100],
                session_id,
            )
            logger.info("Using Agent ARN: %s", AGENT_ARN)

            # Call the AgentCore runtime via ARN
            agent_response = await call_agentcore_runtime(prompt, session_id)

            # Return successful response
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                },
                "body": json.dumps(
                    {
                        "response": agent_response,
                        "sessionId": session_id,
                        "timestamp": context.aws_request_id,
                        "user": user_info.get("email", user_info.get("sub")),
                    }
                ),
            }

    except Exception as e:
        logger.error("Error processing request: %s", e, exc_info=True)
        return create_error_response(500, f"Internal server error: {e}")


def extract_user_from_context(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract user information from API Gateway request context.
    (API Gateway Cognito Authorizer populates this).
    """
    try:
        # Log the full request context for debugging
        request_context = event.get("requestContext", {})
        logger.info("Request context keys: %s", list(request_context.keys()))
        
        # Check authorizer claims
        authorizer = request_context.get("authorizer", {})
        logger.info("Authorizer keys: %s", list(authorizer.keys()))
        
        claims = authorizer.get("claims", {})
        if claims:
            email = claims.get("email")
            sub = claims.get("sub")
            logger.info("✅ User authenticated - Email: %s, Sub: %s", email, sub)
            logger.info("Available claims: %s", list(claims.keys()))
            return claims

        # Check if user info is in a different location
        identity = request_context.get("identity", {})
        logger.info("Identity keys: %s", list(identity.keys()))
        
        cognito_identity_id = identity.get("cognitoIdentityId")
        if cognito_identity_id:
            logger.info("Found Cognito Identity ID: %s", cognito_identity_id)
            return {"sub": cognito_identity_id}

        # Log the full event structure for debugging
        logger.warning("❌ No user information found in request context")
        logger.info("Full event keys: %s", list(event.keys()))
        return {"sub": "unknown", "email": "unknown"}

    except Exception as e:
        logger.error("Error extracting user from context: %s", e)
        return {"sub": "unknown", "email": "unknown"}


def call_agentcore_runtime_sync(prompt: str, session_id: str) -> str:
    """Synchronous call to the AgentCore runtime via ARN."""
    try:
        # Prepare the payload for AgentCore
        payload = json.dumps({"prompt": prompt, "sessionId": session_id}).encode(
            "utf-8"
        )
        logger.info("Payload prepared: %d bytes", len(payload))

        # Generate a random trace ID (keep it short to avoid AgentCore issues)
        trace_id = str(uuid.uuid4())[:8]

        # Invoke the AgentCore service
        logger.info("Invoking AgentCore service with traceId: %s", trace_id)
        response = agent_core_client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_ARN,
            traceId=trace_id,
            runtimeSessionId=session_id,
            payload=payload,
        )
        logger.info("AgentCore response received: %s", type(response))

        # Process the response
        return process_agentcore_response(response)

    except Exception as e:
        logger.error("Error calling AgentCore runtime: %s", e)
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


def call_agentcore_history_sync(session_id: str, k: int) -> Dict[str, Any]:
    """Synchronous call to get conversation history from AgentCore runtime."""
    try:
        # Prepare the payload for history request
        payload = json.dumps({
            "action": "getHistory",
            "sessionId": session_id,
            "k": k
        }).encode("utf-8")
        
        logger.info("History payload prepared: %d bytes", len(payload))

        # Generate a random trace ID
        trace_id = str(uuid.uuid4())[:8]

        # Invoke the AgentCore service for history
        logger.info("Invoking AgentCore service for history with traceId: %s", trace_id)
        response = agent_core_client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_ARN,
            traceId=trace_id,
            runtimeSessionId=session_id,
            payload=payload,
        )
        logger.info("AgentCore history response received: %s", type(response))

        # Process the response
        history_text = process_agentcore_response(response)
        
        # Try to parse as JSON if it looks like JSON
        try:
            if history_text.strip().startswith('{'):
                history_data = json.loads(history_text)
                return history_data
        except json.JSONDecodeError:
            pass
        
        # If not JSON, return empty history
        logger.warning("History response was not JSON: %s", history_text[:100])
        return {"messages": [], "sessionId": session_id}

    except Exception as e:
        logger.error("Error calling AgentCore runtime for history: %s", e)
        return {"messages": [], "sessionId": session_id, "error": str(e)}


async def call_agentcore_history(session_id: str, k: int) -> Dict[str, Any]:
    """
    Async wrapper for the synchronous history call.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(
            executor, call_agentcore_history_sync, session_id, k
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
    This function handles multiple possible content types from the service.
    """
    try:
        content_type = response.get("contentType", "")
        logger.info("Processing response with content type: %s", content_type)
        streaming_body = response.get("response")

        if not streaming_body:
            return ""

        # Case 1: Handle text/event-stream responses (most common for AgentCore)
        if "text/event-stream" in content_type:
            content = []
            if hasattr(streaming_body, "iter_lines"):
                for line in streaming_body.iter_lines(chunk_size=10):
                    if line:
                        decoded_line = line.decode("utf-8")
                        # Strip the "data: " prefix if present
                        if decoded_line.startswith("data: "):
                            content.append(decoded_line[6:])
                        else:
                            content.append(decoded_line)
                return "\n".join(content)

        # Case 2: Handle application/json responses
        elif content_type == "application/json":
            if hasattr(streaming_body, "read"):
                json_content = streaming_body.read().decode("utf-8")
                parsed = json.loads(json_content)
                return parsed.get("response", str(parsed))

        # Case 3: Fallback for other content types or if stream handling fails
        if hasattr(streaming_body, "read"):
            return streaming_body.read().decode("utf-8")
        if hasattr(streaming_body, "iter_lines"):
            return "\n".join(
                [line.decode("utf-8") for line in streaming_body.iter_lines() if line]
            )

        # Final fallback if body is not readable in a standard way
        return str(streaming_body)

    except Exception as e:
        logger.error("Error processing AgentCore response: %s", e, exc_info=True)
        return f"Error processing response: {e}"


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

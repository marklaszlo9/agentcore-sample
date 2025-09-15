"""
AWS Lambda function to proxy requests to AgentCore service
This keeps AWS credentials server-side and provides a clean API for the frontend
"""

import json
import logging
import os
import uuid
from typing import Any, Dict

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize the Bedrock AgentCore client
agent_core_client = boto3.client("bedrock-agentcore")

# AgentCore configuration - get from environment variables
AGENT_ARN = os.environ.get(
    "AGENT_ARN",
    "arn:aws:bedrock-agentcore:us-east-1:886436945166:runtime/hosted_agent_mo6qq-qoks2s8WqG",
)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for AgentCore proxy requests
    """
    try:
        # Handle CORS preflight requests
        if event.get("httpMethod") == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    "Access-Control-Max-Age": "86400",
                },
                "body": "",
            }

        # Parse the request body
        if "body" not in event:
            return create_error_response(400, "Missing request body")

        try:
            if event.get("isBase64Encoded", False):
                import base64

                body = json.loads(base64.b64decode(event["body"]).decode("utf-8"))
            else:
                body = json.loads(event["body"])
        except json.JSONDecodeError as e:
            return create_error_response(400, f"Invalid JSON in request body: {str(e)}")

        # Check if this is a history request
        action = body.get("action", "query")
        session_id = body.get("sessionId", "")

        if action == "getHistory":
            # Handle history request
            if not session_id:
                return create_error_response(400, "Missing 'sessionId' for history request")

            logger.info(f"Processing history request for SessionId: {session_id}")

            # Prepare the payload for AgentCore to get history
            payload = json.dumps({
                "action": "getHistory",
                "sessionId": session_id,
                "k": body.get("k", 3)  # Number of messages to retrieve
            }).encode("utf-8")

        else:
            # Handle regular query request
            prompt = body.get("prompt", "")

            if not prompt:
                return create_error_response(400, "Missing 'prompt' in request body")

            logger.info(
                f"Processing query request - Prompt: {prompt[:100]}..., SessionId: {session_id}"
            )

            # Prepare the payload for AgentCore
            payload = json.dumps({"prompt": prompt, "sessionId": session_id}).encode(
                "utf-8"
            )

        logger.info(f"Using Agent ARN: {AGENT_ARN}")

        logger.info(f"Payload prepared: {len(payload)} bytes")

        # Generate a random trace ID (keep it short to avoid AgentCore issues)
        trace_id = str(uuid.uuid4())[:8]

        # Invoke the AgentCore service
        logger.info(f"Invoking AgentCore service with traceId: {trace_id}")

        response = agent_core_client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_ARN,
            traceId=trace_id,
            runtimeSessionId=session_id,
            payload=payload,
        )

        logger.info(f"AgentCore response received: {type(response)}")

        # Process the response
        if action == "getHistory":
            # For history requests, expect JSON response with messages
            agent_response = process_agentcore_response(response)

            # Try to parse the response as JSON to extract messages
            try:
                if isinstance(agent_response, str):
                    response_data = json.loads(agent_response)
                else:
                    response_data = agent_response

                # Return the messages directly
                return {
                    "statusCode": 200,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "POST, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    },
                    "body": json.dumps({
                        "messages": response_data.get("messages", []),
                        "sessionId": session_id,
                    }),
                }
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Failed to parse history response: {str(e)}")
                # Return empty messages if parsing fails
                return {
                    "statusCode": 200,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "POST, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    },
                    "body": json.dumps({
                        "messages": [],
                        "sessionId": session_id,
                        "error": f"Failed to parse history: {str(e)}"
                    }),
                }
        else:
            # For regular queries, return text response
            agent_response = process_agentcore_response(response)

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
                    }
                ),
            }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
            "body": json.dumps(
                {"error": f"Internal server error: {str(e)}", "statusCode": 500}
            ),
        }


def process_agentcore_response(response: Dict[str, Any]) -> str:
    """
    Process the AgentCore response and extract the text content from StreamingBody
    """
    try:
        logger.info(f"Processing response with keys: {list(response.keys())}")
        content_type = response.get("contentType", "")
        logger.info(f"Content type: {content_type}")

        # Handle text/event-stream responses (most common for AgentCore)
        if "text/event-stream" in content_type:
            logger.info("Processing event-stream response")
            content = []

            # Get the StreamingBody from the response
            streaming_body = response.get("response")
            if streaming_body and hasattr(streaming_body, "iter_lines"):
                try:
                    for line in streaming_body.iter_lines(chunk_size=10):
                        if line:
                            line = line.decode("utf-8")
                            if line.startswith("data: "):
                                line = line[6:]  # Remove "data: " prefix
                            content.append(line)

                    result = "\n".join(content)
                    logger.info(
                        f"Processed event-stream response: {len(result)} characters"
                    )
                    return result

                except Exception as stream_error:
                    logger.error(f"Error reading event stream: {str(stream_error)}")
                    # Fallback to reading the entire stream
                    if hasattr(streaming_body, "read"):
                        content = streaming_body.read()
                        if isinstance(content, bytes):
                            content = content.decode("utf-8")
                        return content
                    return str(streaming_body)

        # Handle application/json responses
        elif content_type == "application/json":
            logger.info("Processing JSON response")
            content = []

            streaming_body = response.get("response")
            if streaming_body:
                try:
                    if hasattr(streaming_body, "iter_lines"):
                        for line in streaming_body.iter_lines():
                            if line:
                                content.append(line.decode("utf-8"))
                    elif hasattr(streaming_body, "read"):
                        content_bytes = streaming_body.read()
                        content.append(content_bytes.decode("utf-8"))
                    else:
                        content.append(str(streaming_body))

                    json_content = "".join(content)
                    parsed_response = json.loads(json_content)
                    logger.info(
                        f"Processed JSON response: {len(json_content)} characters"
                    )
                    return parsed_response.get("response", str(parsed_response))

                except Exception as json_error:
                    logger.error(f"Error processing JSON response: {str(json_error)}")
                    return "".join(content) if content else str(streaming_body)

        # Handle other content types or fallback
        else:
            logger.info(f"Processing response with content type: {content_type}")
            streaming_body = response.get("response")

            if streaming_body:
                try:
                    if hasattr(streaming_body, "read"):
                        content = streaming_body.read()
                        if isinstance(content, bytes):
                            content = content.decode("utf-8")
                        return content
                    elif hasattr(streaming_body, "iter_lines"):
                        content = []
                        for line in streaming_body.iter_lines():
                            if line:
                                content.append(line.decode("utf-8"))
                        return "\n".join(content)
                    else:
                        return str(streaming_body)

                except Exception as fallback_error:
                    logger.error(f"Error in fallback processing: {str(fallback_error)}")
                    return str(streaming_body)

            # Final fallback
            return str(response)

    except Exception as e:
        logger.error(f"Error processing AgentCore response: {str(e)}", exc_info=True)
        return f"Error processing response: {str(e)}"


def create_error_response(status_code: int, message: str) -> Dict[str, Any]:
    """
    Create a standardized error response
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

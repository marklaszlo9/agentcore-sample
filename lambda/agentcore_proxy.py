"""
AWS Lambda function to proxy requests to AgentCore service
This keeps AWS credentials server-side and provides a clean API for the frontend
"""

import json
import logging
import os
import uuid
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor
import asyncio

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
    """Lambda handler that calls the async handler"""
    return asyncio.run(async_lambda_handler(event, context))


async def async_lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
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

        # Extract user info from API Gateway context (already verified by Cognito Authorizer)
        user_info = extract_user_from_context(event)

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

        # Extract the prompt from the request
        prompt = body.get("prompt", "")
        session_id = body.get("sessionId", "")

        if not prompt:
            return create_error_response(400, "Missing 'prompt' in request body")

        logger.info(
            f"Processing request - User: {user_info.get('email', user_info.get('sub', 'unknown'))}, Prompt: {prompt[:100]}..., SessionId: {session_id}"
        )
        logger.info(f"Using Agent ARN: {AGENT_ARN}")

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
                    "user": user_info.get('email', user_info.get('sub')),
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





def extract_user_from_context(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract user information from API Gateway request context
    (API Gateway Cognito Authorizer populates this)
    """
    try:
        # API Gateway puts Cognito user info in requestContext.authorizer.claims
        request_context = event.get("requestContext", {})
        authorizer = request_context.get("authorizer", {})
        claims = authorizer.get("claims", {})
        
        if claims:
            logger.info(f"User authenticated: {claims.get('email', claims.get('sub', 'unknown'))}")
            return claims
        
        # Fallback: try to get from other locations in the context
        cognito_identity_id = request_context.get("identity", {}).get("cognitoIdentityId")
        if cognito_identity_id:
            return {"sub": cognito_identity_id}
            
        logger.warning("No user information found in request context")
        return {"sub": "unknown"}
        
    except Exception as e:
        logger.error(f"Error extracting user from context: {str(e)}")
        return {"sub": "unknown"}


def call_agentcore_runtime_sync(prompt: str, session_id: str) -> str:
    """Call the AgentCore runtime via ARN"""
    try:
        # Prepare the payload for AgentCore
        payload = json.dumps({"prompt": prompt, "sessionId": session_id}).encode("utf-8")
        
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
        agent_response = process_agentcore_response(response)
        return agent_response
                
    except Exception as e:
        logger.error(f"Error calling AgentCore runtime: {e}")
        return get_fallback_response(prompt)


async def call_agentcore_runtime(prompt: str, session_id: str) -> str:
    """Async wrapper for the synchronous runtime call"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, call_agentcore_runtime_sync, prompt, session_id)


def get_fallback_response(prompt: str) -> str:
    """Provide a fallback response when the containerized runtime is not available"""
    
    # Simple keyword-based routing for fallback
    prompt_lower = prompt.lower()
    
    if any(keyword in prompt_lower for keyword in ["envision", "credit", "category", "scoring", "assessment"]):
        return f"""Thank you for your question about the Envision Sustainable Infrastructure Framework.

Your question: "{prompt}"

I'm currently operating in simplified mode. Here's some general guidance about the Envision framework:

**Envision Framework Overview:**
The Envision framework includes five main categories:

1. **Quality of Life (QL)** - Improve community quality of life
2. **Leadership (LD)** - Provide effective leadership and commitment  
3. **Resource Allocation (RA)** - Allocate material and energy resources efficiently
4. **Natural World (NW)** - Protect and restore the natural world
5. **Climate and Resilience (CR)** - Adapt to changing conditions and prepare for long-term resilience

Each category contains specific credits that projects can pursue for sustainability recognition. The framework helps infrastructure projects achieve higher levels of sustainability and resilience.

For detailed information about specific credits and implementation guidance, please try again later when the full system is available."""

    else:
        return f"""Thank you for your sustainability question.

Your question: "{prompt}"

I'm currently operating in simplified mode. Here's some general sustainability guidance:

**General Sustainability Principles:**
- Consider environmental, social, and economic impacts in your decision-making
- Look for opportunities to reduce resource consumption and waste
- Implement renewable energy and efficient systems where possible
- Engage stakeholders and communities in sustainable development
- Follow circular economy principles to minimize waste
- Consider life-cycle impacts of materials and processes

**Key Frameworks to Consider:**
- UN Sustainable Development Goals (SDGs)
- LEED Green Building Standards
- BREEAM Environmental Assessment
- ISO 14001 Environmental Management
- GRI Sustainability Reporting Standards

For more detailed and personalized guidance, please try again later when the full AI system is available."""


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

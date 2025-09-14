FROM public.ecr.aws/docker/library/python:3.13-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --upgrade



# Copy application files
COPY custom_agent.py .
COPY agent_cli.py .
COPY runtime_agent_main.py .
COPY agent_logging.py .
COPY multi_agent_orchestrator.py .


# Create necessary directories
RUN mkdir -p /app/logs

# Expose port
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST="0.0.0.0"



# AgentCore Memory ID should be provided at runtime
# ENV AGENTCORE_MEMORY_ID="your-memory-id"

# Health check - Now implements AgentCore service contract
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the AgentCore runtime with observability (as per AWS docs)
# This follows the bedrock_agentcore_starter_toolkit pattern
CMD ["opentelemetry-instrument", "python", "runtime_agent_main.py"]


# Envision Multi-Agent Architecture Overview

This document describes the corrected architecture where the multi-agent system runs in a containerized AgentCore runtime, not in the Lambda function.

## Architecture Diagram

```
Frontend (CloudFront + S3)
    ↓ (HTTPS + JWT)
API Gateway + Cognito Authorizer
    ↓ (Authenticated requests)
Lambda Proxy Function
    ↓ (HTTP calls)
Containerized AgentCore Runtime
    ↓ (Multi-agent orchestration)
[Orchestrator] → [Knowledge Agent] | [General Agent]
    ↓ (Strands + AgentCore)
Bedrock Models + Memory
```

## Components

### 1. Frontend Layer
- **Static Website**: React/HTML hosted on S3 + CloudFront
- **Authentication**: Cognito-based JWT authentication
- **CORS**: Proper CORS handling for authenticated requests

### 2. API Gateway + Lambda
- **API Gateway**: Routes requests with Cognito authorization
- **Lambda Function**: Simple proxy that calls the AgentCore runtime
- **Purpose**: Keeps AWS credentials secure, handles CORS

### 3. AgentCore Runtime (Container)
- **Technology**: FastAPI + Docker container
- **Multi-Agent System**: Strands agents with AgentCore integration
- **Orchestrator**: Routes queries to appropriate specialist agents
- **Memory**: Persistent conversation memory via AgentCore MemoryClient

### 4. Agent Specialists
- **Knowledge Agent**: Envision Framework expertise
- **General Agent**: Broad sustainability knowledge
- **Hooks**: Context enhancement and intelligent routing

## File Structure

```
├── Dockerfile                   # Container definition
├── runtime_agent_main.py       # AgentCore runtime with multi-agent support
├── multi_agent_orchestrator.py # Multi-agent orchestration logic
├── custom_agent.py             # Single agent fallback
├── requirements.txt             # Container dependencies
├── lambda/                      # Lambda proxy function
│   └── agentcore_proxy.py       # Calls containerized runtime
├── static-frontend/             # Frontend application
├── infrastructure/              # CloudFormation templates
└── deploy-frontend.sh           # Deployment script
```

## Deployment Process

### Step 1: Build and Deploy AgentCore Runtime
```bash
# Build Docker image
docker build -t envision-agentcore .

# Tag and push to ECR (replace with your ECR URI)
docker tag envision-agentcore:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/envision-agentcore:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/envision-agentcore:latest
```

This will:
1. Build Docker image with multi-agent system
2. Push to ECR
3. Ready for container deployment (ECS/EKS/Fargate)

### Step 2: Deploy Frontend Infrastructure
```bash
./deploy-frontend.sh [stack-name] [runtime-url] [cognito-pool] [cognito-client]
```

Example:
```bash
./deploy-frontend.sh envision-frontend http://agentcore-runtime:8080 us-east-1_ABC123 client123
```

## Benefits of This Architecture

### ✅ Separation of Concerns
- **Lambda**: Simple proxy, fast cold starts
- **Container**: Complex multi-agent logic, persistent connections

### ✅ Scalability
- **Lambda**: Scales automatically for API requests
- **Container**: Can be scaled independently based on AI workload

### ✅ Development
- **Container**: Easy local development and testing
- **Lambda**: Simple proxy logic, fewer dependencies

### ✅ Cost Optimization
- **Lambda**: Pay per request for API calls
- **Container**: Optimized for AI workload, can use spot instances

## Local Development

### Run AgentCore Runtime Locally
```bash
pip install -r requirements.txt
python runtime_agent_main.py
```

Or with Docker:
```bash
docker build -t envision-agentcore .
docker run -p 8080:8080 -e USE_MULTI_AGENT=true envision-agentcore
```

### Test the Runtime
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are Quality of Life credits?", "sessionId": "test-123"}'
```

### Test Lambda Locally
Set environment variable:
```bash
export AGENTCORE_RUNTIME_URL=http://localhost:8080
```

## Production Deployment Options

### Option 1: ECS Fargate
- Serverless containers
- Auto-scaling based on CPU/memory
- Integrated with ALB for load balancing

### Option 2: EKS
- Kubernetes orchestration
- Advanced scaling and deployment strategies
- Service mesh integration

### Option 3: ECS EC2
- More control over underlying infrastructure
- Cost optimization with reserved instances
- Custom AMIs and configurations

## Monitoring and Observability

### AgentCore Runtime
- **Health Check**: `/health` endpoint
- **Metrics**: FastAPI built-in metrics
- **Logs**: Structured JSON logging
- **Tracing**: OpenTelemetry integration

### Lambda Function
- **CloudWatch Logs**: Request/response logging
- **X-Ray Tracing**: Request tracing
- **Metrics**: Duration, errors, throttles

## Security Considerations

### Network Security
- **VPC**: Deploy container in private subnets
- **Security Groups**: Restrict access to Lambda only
- **NAT Gateway**: For outbound internet access

### Authentication
- **Cognito**: JWT token validation at API Gateway
- **IAM**: Lambda execution role with minimal permissions
- **Container**: No direct internet access required

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Check API Gateway CORS configuration
   - Verify Lambda returns proper headers

2. **Container Connection Issues**
   - Verify runtime URL in Lambda environment
   - Check security group rules
   - Test container health endpoint

3. **Authentication Failures**
   - Verify Cognito configuration
   - Check JWT token format
   - Validate API Gateway authorizer

### Debugging Steps

1. **Check Container Logs**
   ```bash
   docker logs [container-id]
   ```

2. **Test Runtime Directly**
   ```bash
   curl http://runtime-url/health
   ```

3. **Check Lambda Logs**
   ```bash
   aws logs tail /aws/lambda/function-name --follow
   ```

This architecture provides a clean separation between the API layer (Lambda) and the AI processing layer (containerized runtime), making the system more maintainable, scalable, and cost-effective.
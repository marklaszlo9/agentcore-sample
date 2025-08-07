# Envision Framework Agent

A serverless AI agent built using AWS Bedrock AgentCore, designed to help users understand and apply the Envision Sustainable Infrastructure Framework. The solution features a static frontend deployed on CloudFront with a Lambda proxy for secure AgentCore integration.

## Architecture

### Serverless Frontend Architecture
- **Static Frontend**: HTML/CSS/JavaScript hosted on S3 and served via CloudFront
- **Lambda Proxy**: Serverless function that securely calls AgentCore APIs
- **API Gateway**: RESTful endpoints with CORS support
- **AgentCore Integration**: AWS Bedrock AgentCore for AI capabilities with persistent memory

### Components
- `static-frontend/index.html`: Chat interface for user interactions
- `lambda/agentcore_proxy.py`: Lambda function for AgentCore integration
- `infrastructure/frontend-infrastructure.yaml`: CloudFormation template for AWS resources
- `custom_agent.py`: Core agent implementation extending AgentCore
- `agent_cli.py`: Command-line interface for local testing

## Features

- **Serverless Architecture**: No servers to manage, scales automatically
- **Secure API Access**: AWS credentials stay server-side in Lambda
- **CORS Support**: Proper cross-origin resource sharing for web frontend
- **Memory Management**: Persistent conversation memory using AgentCore
- **CloudFront CDN**: Global content delivery for fast frontend loading
- **Infrastructure as Code**: Complete AWS infrastructure defined in CloudFormation

## Deployment

### Prerequisites
- AWS CLI configured with appropriate permissions
- AWS account with access to:
  - S3, CloudFront, Lambda, API Gateway
  - AWS Bedrock AgentCore

### Deploy the Complete Solution

```bash
# Make deployment script executable
chmod +x deploy-frontend.sh

# Deploy everything (infrastructure + frontend + Lambda)
./deploy-frontend.sh [stack-name] [agent-runtime-arn]

# Example:
./deploy-frontend.sh envision-agent-frontend arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/your-agent
```

### Update Lambda Function Only

```bash
# Make update script executable
chmod +x update-lambda.sh

# Update just the Lambda function code
./update-lambda.sh [function-name] [handler-file]

# Example:
./update-lambda.sh envision-agent-frontend-agentcore-proxy agentcore_proxy.py
```

## Usage

### Web Interface
After deployment, access the frontend via the CloudFront URL provided in the deployment output. The chat interface allows natural language interaction with the Envision Framework agent.

### CLI Interface (Local Testing)
```bash
python agent_cli.py
```

### API Testing
```bash
# Test the Lambda proxy directly
curl -X POST https://your-api-gateway-url/prod/agentcore \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the key principles of sustainable infrastructure?"}'
```

## Configuration

### Environment Variables (Lambda)
- `AGENT_ARN`: AgentCore runtime ARN (set automatically by CloudFormation)

### Frontend Configuration
Update the API URL in `static-frontend/index.html`:
```javascript
const LAMBDA_API_URL = 'https://your-api-gateway-url/prod/agentcore';
```

## Development

### Local Development
For local testing of the agent logic:
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AGENTCORE_MEMORY_ID=your-memory-id
export AWS_DEFAULT_REGION=us-east-1

# Run CLI interface
python agent_cli.py
```

## Project Structure

```
├── static-frontend/           # Frontend web application
│   └── index.html            # Chat interface
├── lambda/                   # Lambda functions
│   ├── agentcore_proxy.py   # Main Lambda proxy function
│   └── simple_proxy.py      # Simplified test function
├── infrastructure/          # AWS infrastructure
│   ├── frontend-infrastructure.yaml  # CloudFormation template
│   └── lambda-requirements.txt       # Lambda dependencies
├── custom_agent.py         # Core agent implementation
├── agent_cli.py           # CLI interface
├── deploy-frontend.sh     # Complete deployment script
├── update-lambda.sh       # Lambda update script
└── requirements.txt       # Python dependencies
```

## Troubleshooting

### CORS Issues
- Ensure Lambda function returns proper CORS headers
- Check API Gateway configuration
- Verify CloudFront origin settings

### Lambda Function Issues
- Check CloudWatch logs for detailed error messages
- Verify IAM permissions for AgentCore access
- Ensure traceId is properly formatted (short UUID)

### Frontend Issues
- Verify API Gateway URL in frontend configuration
- Check browser developer tools for network errors
- Ensure CloudFront distribution is properly deployed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally and with deployment
5. Submit a pull request

## License

[Add your license information here]
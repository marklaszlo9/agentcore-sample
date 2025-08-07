# Frontend Deployment Guide: S3 + CloudFront + Lambda

This guide shows how to deploy the Envision Agent frontend to AWS using S3 + CloudFront for the static frontend and Lambda as a proxy to your AgentCore service.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │
│   Browser   │───▶│ CloudFront  │───▶│   Lambda    │───▶│ AgentCore   │
│             │◀───│   (CDN)     │◀───│   (Proxy)   │◀───│  Service    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │             │
                   │ S3 Bucket   │
                   │ (Frontend)  │
                   └─────────────┘
```

## Benefits

✅ **Performance**: CloudFront CDN for global distribution  
✅ **Security**: AWS credentials stay server-side in Lambda  
✅ **Scalability**: Serverless architecture scales automatically  
✅ **Cost-Effective**: Pay only for what you use  
✅ **CORS Handling**: Lambda handles CORS automatically  
✅ **SSL/TLS**: HTTPS by default with CloudFront  

## Prerequisites

1. **AWS CLI** configured with appropriate permissions
2. **AgentCore service** deployed and running
3. **Agent Runtime ARN** from your AgentCore deployment

### Required IAM Permissions

Your AWS user/role needs these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "cloudformation:*",
                "s3:*",
                "cloudfront:*",
                "lambda:*",
                "apigateway:*",
                "iam:CreateRole",
                "iam:DeleteRole",
                "iam:AttachRolePolicy",
                "iam:DetachRolePolicy",
                "iam:PassRole",
                "bedrock-agentcore:InvokeAgentRuntime"
            ],
            "Resource": "*"
        }
    ]
}
```

## Quick Deployment

### Option 1: Automated Deployment (Recommended)

```bash
# Make sure you're in the project root directory
cd /path/to/your/project

# Run the deployment script
./deploy-frontend.sh greencross-frontend-886436945166 arn:aws:bedrock-agentcore:us-east-1:886436945166:runtime/hosted_agent_mo6qq-qoks2s8WqG

# The script will:
# 1. Package the Lambda function
# 2. Deploy CloudFormation stack
# 3. Update Lambda code
# 4. Configure frontend with API Gateway URL
# 5. Upload frontend to S3
# 6. Invalidate CloudFront cache
```

### Option 2: Manual Deployment

```bash
# 1. Package Lambda function
cd lambda
zip -r ../agentcore-proxy.zip agentcore_proxy.py
cd ..

# 2. Deploy CloudFormation stack
aws cloudformation deploy \
    --template-file infrastructure/frontend-infrastructure.yaml \
    --stack-name envision-agent-frontend \
    --parameter-overrides \
        AgentRuntimeArn="arn:aws:bedrock-agentcore:us-east-1:886436945166:runtime/hosted_agent_mo6qq-qoks2s8WqG" \
    --capabilities CAPABILITY_IAM

# 3. Get stack outputs
aws cloudformation describe-stacks \
    --stack-name envision-agent-frontend \
    --query 'Stacks[0].Outputs'

# 4. Update Lambda function code
aws lambda update-function-code \
    --function-name envision-agent-frontend-agentcore-proxy \
    --zip-file fileb://agentcore-proxy.zip

# 5. Update frontend configuration with API Gateway URL
# Edit static-frontend/index.html and update LAMBDA_API_URL

# 6. Upload frontend to S3
aws s3 sync static-frontend/ s3://your-bucket-name/ --delete

# 7. Invalidate CloudFront cache
aws cloudfront create-invalidation \
    --distribution-id YOUR_DISTRIBUTION_ID \
    --paths "/*"
```

## Configuration

### Frontend Configuration

Update `static-frontend/index.html`:

```javascript
// Configuration
const LAMBDA_API_URL = 'https://your-api-gateway-url.amazonaws.com/prod/agentcore';
const USE_LAMBDA = true; // Always true for this deployment
```

### Lambda Configuration

The Lambda function is configured via environment variables:

- `AGENT_ARN`: Your AgentCore runtime ARN

## Testing

### Test Lambda Proxy

```bash
# Test the Lambda proxy directly
curl -X POST https://your-api-gateway-url.amazonaws.com/prod/agentcore \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the Envision framework?"}'

# Expected response:
{
  "response": "The Envision framework is...",
  "sessionId": "session_abc123",
  "timestamp": "request-id-123"
}
```

### Test Frontend

1. Open your CloudFront URL in a browser
2. Type a message in the chat interface
3. Verify the response comes from your AgentCore service

### Debug Issues

```bash
# Check Lambda logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/envision-agent-frontend"

# View recent Lambda logs
aws logs tail /aws/lambda/envision-agent-frontend-agentcore-proxy --follow

# Check CloudFront distribution status
aws cloudfront get-distribution --id YOUR_DISTRIBUTION_ID

# Test S3 bucket access
aws s3 ls s3://your-bucket-name/
```

## Monitoring and Maintenance

### CloudWatch Metrics

Monitor these key metrics:

- **Lambda**: Invocations, Duration, Errors
- **API Gateway**: Count, Latency, 4XXError, 5XXError
- **CloudFront**: Requests, BytesDownloaded, ErrorRate

### Cost Optimization

1. **CloudFront**: Use appropriate price class
2. **Lambda**: Optimize memory allocation
3. **S3**: Use appropriate storage class
4. **API Gateway**: Consider caching for repeated requests

### Updates

To update the frontend:

```bash
# Update frontend files
# Edit static-frontend/index.html

# Re-run deployment
./deploy-frontend.sh envision-agent-frontend
```

To update Lambda function:

```bash
# Edit lambda/agentcore_proxy.py
# Re-run deployment
./deploy-frontend.sh envision-agent-frontend
```

## Security Considerations

### Lambda Security

- ✅ **IAM Role**: Minimal permissions for AgentCore access only
- ✅ **Environment Variables**: Sensitive config stored securely
- ✅ **VPC**: Can be deployed in VPC if needed
- ✅ **Encryption**: All data encrypted in transit and at rest

### Frontend Security

- ✅ **HTTPS**: Enforced by CloudFront
- ✅ **CORS**: Properly configured in Lambda
- ✅ **CSP**: Can be added via CloudFront headers
- ✅ **No Credentials**: AWS credentials never exposed to browser

### API Gateway Security

- ✅ **Rate Limiting**: Built-in throttling
- ✅ **CORS**: Configured for browser access
- ✅ **Logging**: Request/response logging available
- ✅ **Monitoring**: CloudWatch integration

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Check Lambda CORS headers
   - Verify API Gateway CORS configuration
   - Test OPTIONS preflight requests

2. **Lambda Timeout**
   - Increase Lambda timeout (max 15 minutes)
   - Optimize AgentCore response processing
   - Add retry logic for transient failures

3. **CloudFront Caching Issues**
   - Invalidate cache after updates
   - Check cache behaviors
   - Verify S3 bucket policy

4. **AgentCore Connection Issues**
   - Verify Agent Runtime ARN
   - Check IAM permissions
   - Test AgentCore service directly

### Debug Commands

```bash
# Test Lambda function locally (if using SAM)
sam local invoke AgentCoreProxyFunction -e test-event.json

# Check API Gateway logs
aws logs tail /aws/apigateway/envision-agent-frontend-api --follow

# Validate CloudFormation template
aws cloudformation validate-template \
    --template-body file://infrastructure/frontend-infrastructure.yaml
```

## Cleanup

To remove all resources:

```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name envision-agent-frontend

# Empty and delete S3 bucket (if needed)
aws s3 rm s3://your-bucket-name --recursive
aws s3 rb s3://your-bucket-name
```

## Custom Domain (Optional)

To use a custom domain:

1. **Get SSL Certificate** in ACM (us-east-1 for CloudFront)
2. **Update CloudFormation** with domain parameters:

```bash
aws cloudformation deploy \
    --template-file infrastructure/frontend-infrastructure.yaml \
    --stack-name envision-agent-frontend \
    --parameter-overrides \
        AgentRuntimeArn="your-agent-arn" \
        DomainName="chat.yourdomain.com" \
        CertificateArn="arn:aws:acm:us-east-1:123456789012:certificate/abc123"
```

3. **Update DNS** to point to CloudFront distribution

## Support

For issues with this deployment:

1. Check CloudWatch logs for Lambda and API Gateway
2. Verify AgentCore service is running
3. Test each component individually
4. Review IAM permissions

Your frontend is now deployed with enterprise-grade infrastructure! 🚀
#!/bin/bash

# Deploy Envision Agent Frontend to S3 + CloudFront + Lambda with Cognito Authentication
# Usage: ./deploy-frontend.sh [stack-name] [agent-runtime-arn] [cognito-user-pool-id] [cognito-client-id]

set -e

# Configuration
STACK_NAME=${1:-"envision-agent-frontend"}
AGENT_RUNTIME_ARN=${2:-"arn:aws:bedrock-agentcore:us-east-1:886436945166:runtime/hosted_agent_sample-KEQNVq8Whv"}
COGNITO_USER_POOL_ID=${3:-"us-east-1_O6RLwBTHN"}
COGNITO_CLIENT_ID=${4:-"734547mm50iohgjf1is1oa36qc"}
REGION=${AWS_DEFAULT_REGION:-"us-east-1"}

echo "🚀 Deploying Envision Agent Frontend with Cognito Authentication"
echo "=============================================================="
echo "Stack Name: $STACK_NAME"
echo "Agent Runtime ARN: $AGENT_RUNTIME_ARN"
echo "Cognito User Pool ID: $COGNITO_USER_POOL_ID"
echo "Cognito Client ID: $COGNITO_CLIENT_ID"
echo "Region: $REGION"
echo ""

# Step 1: Package Lambda function
echo "📦 Step 1: Packaging Lambda function..."
cd lambda
zip -r ../agentcore-proxy.zip agentcore_proxy.py
cd ..

# Step 2: Deploy CloudFormation stack
echo "☁️  Step 2: Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file infrastructure/frontend-infrastructure.yaml \
    --stack-name "$STACK_NAME" \
    --parameter-overrides \
        AgentRuntimeArn="$AGENT_RUNTIME_ARN" \
        CognitoUserPoolId="$COGNITO_USER_POOL_ID" \
        CognitoClientId="$COGNITO_CLIENT_ID" \
    --capabilities CAPABILITY_IAM \
    --region "$REGION"

# Step 3: Update Lambda function code
echo "🔧 Step 3: Updating Lambda function code..."
LAMBDA_FUNCTION_NAME=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`LambdaFunctionName`].OutputValue' \
    --output text)

aws lambda update-function-code \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --zip-file fileb://agentcore-proxy.zip \
    --region "$REGION"

# Step 4: Get stack outputs
echo "📋 Step 4: Getting deployment information..."
BUCKET_NAME=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`FrontendBucketName`].OutputValue' \
    --output text)

API_GATEWAY_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiGatewayURL`].OutputValue' \
    --output text)

FRONTEND_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`FrontendURL`].OutputValue' \
    --output text)

CLOUDFRONT_DISTRIBUTION_ID=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`CloudFrontDistributionId`].OutputValue' \
    --output text)

# Step 5: Update frontend configuration
echo "🔧 Step 5: Updating frontend configuration..."

# Update API URL in index.html
sed -i.bak "s|const LAMBDA_API_URL = '.*';|const LAMBDA_API_URL = '$API_GATEWAY_URL';|g" static-frontend/index.html

# Update Cognito configuration in auth.js
echo "🔧 Step 5a: Updating Cognito configuration..."
sed -i.bak "s|redirect_uri: \".*\"|redirect_uri: \"$FRONTEND_URL\"|g" static-frontend/auth.js

# Update cognito-config.json for future reference
if [ -f "cognito-config.json" ]; then
    echo "🔧 Step 5b: Updating cognito-config.json..."
    python3 -c "
import json
with open('cognito-config.json', 'r') as f:
    config = json.load(f)
config['redirectUri'] = '$FRONTEND_URL'
config['logoutUri'] = '$FRONTEND_URL'
config['userPoolId'] = '$COGNITO_USER_POOL_ID'
config['clientId'] = '$COGNITO_CLIENT_ID'
with open('cognito-config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
fi

# Step 6: Upload frontend to S3
echo "📤 Step 6: Uploading frontend to S3..."
aws s3 sync static-frontend/ s3://"$BUCKET_NAME"/ \
    --delete \
    --cache-control "max-age=86400" \
    --region "$REGION"

# Step 7: Invalidate CloudFront cache
echo "🔄 Step 7: Invalidating CloudFront cache..."
aws cloudfront create-invalidation \
    --distribution-id "$CLOUDFRONT_DISTRIBUTION_ID" \
    --paths "/*" \
    --region "$REGION"

# Cleanup
rm -f agentcore-proxy.zip
rm -f static-frontend/index.html.bak
rm -f static-frontend/auth.js.bak

echo ""
echo "✅ Deployment Complete!"
echo "======================"
echo "Frontend URL: $FRONTEND_URL"
echo "API Gateway URL: $API_GATEWAY_URL"
echo "S3 Bucket: $BUCKET_NAME"
echo "CloudFront Distribution ID: $CLOUDFRONT_DISTRIBUTION_ID"
echo ""
echo "🔐 Authentication Configuration:"
echo "User Pool ID: $COGNITO_USER_POOL_ID"
echo "Client ID: $COGNITO_CLIENT_ID"
echo ""
echo "📝 Next Steps:"
echo "1. Update your Cognito App Client settings:"
echo "   - Callback URLs: $FRONTEND_URL"
echo "   - Sign out URLs: $FRONTEND_URL"
echo "2. Ensure your Cognito User Pool domain is configured"
echo "3. Test the authentication flow"
echo ""
echo "🧪 Test authenticated API call (requires valid JWT token):"
echo "curl -X POST $API_GATEWAY_URL \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'Authorization: Bearer <your-jwt-token>' \\"
echo "  -d '{\"prompt\": \"Hello, what can you help me with?\"}'"
echo ""
echo "🌐 Open your frontend: $FRONTEND_URL"
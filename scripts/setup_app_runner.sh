#!/bin/bash
# scripts/setup_app_runner.sh
# Run this ONCE locally to create the App Runner service and IAM role.
# Prerequisites: AWS CLI configured, ECR repo already created, image pushed.
#
# Usage:
#   chmod +x scripts/setup_app_runner.sh
#   ./scripts/setup_app_runner.sh

set -e

REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="msml605-backend"
SERVICE_NAME="msml605-backend"
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest"

echo "Account ID : $ACCOUNT_ID"
echo "Region     : $REGION"
echo "Image URI  : $IMAGE_URI"
echo ""

# ── Step 1: Create IAM role that App Runner uses to pull from ECR ─────────────
echo "Creating IAM role for App Runner ECR access..."

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Service": "build.apprunner.amazonaws.com"
    },
    "Action": "sts:AssumeRole"
  }]
}'

aws iam create-role \
  --role-name AppRunnerECRAccessRole \
  --assume-role-policy-document "$TRUST_POLICY" \
  --region "$REGION" 2>/dev/null || echo "Role already exists, skipping"

aws iam attach-role-policy \
  --role-name AppRunnerECRAccessRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess \
  2>/dev/null || echo "Policy already attached, skipping"

ACCESS_ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/AppRunnerECRAccessRole"
echo "IAM role ARN: $ACCESS_ROLE_ARN"
echo ""

# ── Step 2: Push initial image to ECR ─────────────────────────────────────────
echo "Logging in to ECR and pushing initial image..."

aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin \
  "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker build -t "$IMAGE_URI" .
docker push "$IMAGE_URI"
echo "Image pushed: $IMAGE_URI"
echo ""

# ── Step 3: Create App Runner service ─────────────────────────────────────────
echo "Creating App Runner service..."

SERVICE_CONFIG=$(cat <<EOF
{
  "ServiceName": "$SERVICE_NAME",
  "SourceConfiguration": {
    "ImageRepository": {
      "ImageIdentifier": "$IMAGE_URI",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8000"
      }
    },
    "AuthenticationConfiguration": {
      "AccessRoleArn": "$ACCESS_ROLE_ARN"
    },
    "AutoDeploymentsEnabled": false
  },
  "InstanceConfiguration": {
    "Cpu": "1024",
    "Memory": "2048"
  },
  "AutoScalingConfigurationArn": "",
  "HealthCheckConfiguration": {
    "Protocol": "HTTP",
    "Path": "/api/status",
    "Interval": 10,
    "Timeout": 5,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 5
  },
  "ObservabilityConfiguration": {
    "ObservabilityEnabled": true
  }
}
EOF
)

RESULT=$(aws apprunner create-service \
  --cli-input-json "$SERVICE_CONFIG" \
  --region "$REGION")

SERVICE_ARN=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['Service']['ServiceArn'])")
SERVICE_URL=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['Service']['ServiceUrl'])")

echo ""
echo "App Runner service created successfully!"
echo ""
echo "────────────────────────────────────────────────────────"
echo "SERVICE ARN : $SERVICE_ARN"
echo "SERVICE URL : https://$SERVICE_URL"
echo "────────────────────────────────────────────────────────"
echo ""
echo "Add these to GitHub Secrets now:"
echo "  APP_RUNNER_SERVICE_ARN = $SERVICE_ARN"
echo "  APP_RUNNER_URL         = https://$SERVICE_URL"
echo ""
echo "Waiting for service to reach RUNNING state..."
aws apprunner wait service-running \
  --service-arn "$SERVICE_ARN" \
  --region "$REGION"

echo "Service is RUNNING."
echo "Test it: curl https://$SERVICE_URL/api/status"

# ── Step 4: Configure auto-scaling ────────────────────────────────────────────
echo ""
echo "Configuring auto-scaling (min 1, max 3 instances)..."

SCALING_ARN=$(aws apprunner create-auto-scaling-configuration \
  --auto-scaling-configuration-name msml605-scaling \
  --max-concurrency 100 \
  --min-size 1 \
  --max-size 3 \
  --region "$REGION" \
  --query 'AutoScalingConfiguration.AutoScalingConfigurationArn' \
  --output text)

aws apprunner update-service \
  --service-arn "$SERVICE_ARN" \
  --auto-scaling-configuration-arn "$SCALING_ARN" \
  --region "$REGION" > /dev/null

echo "Auto-scaling configured:"
echo "  Min instances : 1"
echo "  Max instances : 3"
echo "  Max concurrency per instance: 100 requests"
echo ""
echo "Setup complete."
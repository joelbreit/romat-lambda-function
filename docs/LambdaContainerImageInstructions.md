# Lambda Container Image Instructions

## Create a Lambda Function

1. Create function with AWS console
2. Select Container Image as the runtime
3. Select image URI if you have already created it
4. Select arm64 if you are running Docker on an Apple Silicon Mac
5. Provide the function with a role with access to CloudWatch Logs and any other AWS services it needs

## Create a test event

1. Create JSON file
2. Save it as `event.json` locally
3. Add it to the Lambda function

## Create a Dockerfile

4. Install AWS CLI: `brew install awscli`
5. Create a directory for the container image: `mkdir <ecr-image-name>`
6. Add your Lambda function code to the directory created in step 2.
7. Create a Dockerfile based on this example and save it as `Dockerfile` in the directory created in step 2:

```Dockerfile
# Use the Amazon Linux 2 base image for Lambda
FROM public.ecr.aws/lambda/python:3.8

# Install necessary dependencies
RUN yum install -y python3-devel gcc
RUN yum install -y mesa-libGL

RUN pip install --upgrade pip
RUN pip install opencv-python-headless mediapipe numpy boto3

# Copy the function code to the Docker image
COPY main.py ${LAMBDA_TASK_ROOT}
COPY pose.py ${LAMBDA_TASK_ROOT}

# Make sure the scripts are executable
RUN chmod 644 ${LAMBDA_TASK_ROOT}/main.py
RUN chmod 644 ${LAMBDA_TASK_ROOT}/pose.py

# Set environment variable for Matplotlib
ENV MPLCONFIGDIR=/tmp

# Set the CMD to your handler
CMD ["main.lambda_handler"]
```

5. Start the Docker daemon app
6. Build the Docker image (from within the folder which has the Dockerfile etc): `docker build -t <ecr-image-name> .`
7. Get the AWS account ID from the console (top right corner)
8. Tag the image: `docker tag <ecr-image-name>:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/<ecr-image-name>:latest`

## Run the Docker image locally

1. Run the Docker image: `docker run -p 9000:8080 <ecr-image-name>`
2. Run this curl command in another terminal: `curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d @event.json`

## Push the image to the ECR

1.  Generate an AWS access keys:
    1.  AWS Console
    2.  IAM
    3.  Users
    4.  Add user
        1. "cli_user"
        2. Add group
            1. "Admins"
            2. Attached AdministratorAccess policy
        3. Add user to group
    5.  Select the user
    6.  Security credentials
    7.  Create access key, CLI
    8.  Download the CSV
2.  Configure the AWS CLI: `aws configure`
3.  Log in to the ECR: `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com`
4.  Create an ECR Repository: `aws ecr create-repository --repository-name <ecr-image-name>`
5.  Push the image to the ECR: `docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/<ecr-image-name>:latest`

## Attach the ECR image to the Lambda function


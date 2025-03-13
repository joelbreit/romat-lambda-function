# Update Container-Based Lambda Function

1. Start the Docker daemon app
2. Navigate to the code folder: `cd lambda-media-pipeline`
3. Log in to the ECR (needs done once per session): `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com`
4. Update the code locally
5. Build the Docker image: `docker build -t <ecr-image-name> .`
6. Tag the image: `docker tag <ecr-image-name>:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/<ecr-image-name>:latest`
7. Test it locally
   1. Run the Docker image: `docker run -p 9000:8080 <ecr-image-name>`
   2. Run this curl command in another terminal: `curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d @Test.json`
8. Push the image to the ECR: `docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/<ecr-image-name>:latest`
# Process of Resetup

1. Ran aws configure
2. Tried to run the Docker image `docker build -t RoMA-T_AnalyzeVideo .` in the `lambda-mediapipe` directory

Ran into this error:

`zsh: command not found: docker`

I just needed to install Docker Desktop to get the Docker CLI.

2. Installed Docker Desktop

Ran into this error:

`ERROR: invalid tag "RoMA-T_AnalyzeVideo": repository name must be lowercase`

This name is just for the ECR image, not the Lambda function, so changing it to `roma-t_analyzevideo`.

1. Built the Docker image `docker build -t roma-t_analyzevideo .` in the `lambda-mediapipe` directory
2. Tagged the Docker image `docker tag roma-t_analyzevideo:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/roma-t_analyzevideo:latest`
3. Ran the Docker image: `docker run -p 9000:8080 roma-t_analyzevideo`
4. In another terminal, ran a request: `curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d @Test.json`

Ran into this error:

`curl: Failed to open Test.json`

My other terminal was in the wrong directory, so I just needed to move it navigate to the `lambda-mediapipe` directory.

Ran into this error:

`"errorMessage": "Unable to locate credentials"`

Didn't get this fixed yet.

1. Logged into ECR: `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com`
2. Created an ECR Repository: `aws ecr create-repository --repository-name roma-t_analyzevideo`
3. Pushed the image to the ECR: `docker push <ecr-image-name>.dkr.ecr.us-east-1.amazonaws.com/roma-t_analyzevideo:latest`
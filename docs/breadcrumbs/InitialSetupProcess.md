# Lambda Bread Crumbs

## ~~Tried using Lambda Layers~~

### 1. Created a Lambda Function on AWS

1. Went to AWS Lambda
2. Created a new function
3. Used the following settings:
    - Function Name: RoMA-T_AnalyzeVideo
    - Runtime: Python 3.12
    - Architecture: x86_64
    - Role: Use an existing role -> LambdaDynamoDBAndS3FullAccessRole

### 2. Created Zip Files

1. Tried with a single zip file containing all dependencies
    1. Navigated to my desired parent directory
    2. `mkdir python`
    3. `pip install opencv-python-headless mediapipe numpy -t python/`
    4. `zip -r dependencies.zip python`
    5. Zip file was too large
2. Tried breaking media pipe and opencv into separate layers
    1. Navigated to my desired parent directory
    2. `mkdir mediapipe_layer`
    3. `mkdir mediapipe_layer/python`
    4. `pip install mediapipe -t mediapipe_layer/python`
    5. `cd mediapipe_layer`
    6. `zip -r mediapipe_layer.zip python`
    7. `cd ../`
    8. `mkdir opencv_numpy_layer`
    9. `mkdir opencv_numpy_layer/python`
    10. `pip install opencv-python-headless numpy -t opencv_numpy_layer/python`
    11. `cd opencv_numpy_layer`
    12. `zip -r opencv_numpy_layer.zip python`
    13. Still too large
3. Tried removing unnecessary files
    1. `cd mediapipe_layer`
    2. `find . -name "*.so" -exec strip --strip-unneeded {} \;`
    3. `find . -name "test*" -exec rm -rf {} \;`
    4. `find . -name "__pycache__" -exec rm -rf {} \;`
    5. `find . -name "*.md" -exec rm -rf {} \;`
    6. `find . -name "examples" -exec rm -rf {} \;`
    7. `find . -name "*.egg-info" -exec rm -rf {} \;`
    8. `find . -name "tests" -exec rm -rf {} \;`
    9. `find . -name "*.dist-info" -exec rm -rf {} \;`
    10. `cd ../`
    11. `zip -r mediapipe_layer.zip python`
    12. Still too large
4. Tried zipping with maximum compression
    1. `zip -r9 mediapipe_layer.zip python`
    2. Still too large
5. Tried using a smaller version of mediapipe
    1. `pip install mediapipe==0.8.3 -t mediapipe_layer/python`
    2. "Could not find a version that satisfies the requirement mediapipe==0.8.3"

## Tried Using an AWS Lambda container image

1. Installed AWS CLI: `brew install awscli`
2. Created a directory for the container image: `mkdir lambda-mediapipe`
3. Created a Dockerfile (saved as Dockerfile):

```Dockerfile
# Use the Amazon Linux 2 base image for Lambda
FROM public.ecr.aws/lambda/python:3.8

# Install necessary dependencies
RUN yum install -y python3-devel gcc
RUN pip install --upgrade pip
RUN pip install opencv-python-headless mediapipe numpy

# Copy the function code
COPY main.py ${LAMBDA_TASK_ROOT}
COPY pose.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["main.lambda_handler"]
```

2. Added a modified main.py
3. Added pose.py
4. Started the Docker daemon app
5. Built the Docker image: `docker build -t lambda-mediapipe .`
6. Got the AWS account ID from the console (top right corner)
7. Tagged the image: `docker tag lambda-mediapipe:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/lambda-mediapipe:latest`
8. Generated AWS access keys
    1. AWS Console
    2. IAM
    3. Users
    4. Add user
        1. "cli_user"
        2. Add group
            1. "Admins"
            2. Attached AdministratorAccess policy
        3. Add user to group
    5. Select the user
    6. Security credentials
    7. Create access key, CLI
    8. Downloaded the CSV
9. Configured the AWS CLI: `aws configure`
10. Logged in to the ECR: `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com`
11. Created an ECR Repository: `aws ecr create-repository --repository-name lambda-mediapipe`
12. Pushed the image to the ECR: `docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/lambda-mediapipe:latest`
13. Created a Lambda function
    1. AWS Console
    2. Lambda
    3. Created function "RoMA-T_MediaPipe-Function"
    4. Container image
    5. Found the image URI
    6. Used LambdaDynamoDBAndS3FullAccessRole
    7. Created the function
14. Created S3 bucket
    1. AWS Console
    2. S3
    3. Created bucket "roma-t.test"
    4. Added a test video
15. Added test case to Lambda function

```json
{
	"bucket": "roma-t.test",
	"video_key": "Left_4_Sitting.MOV",
	"frame_step": 10,
	"angles": ["left_knee"]
}
```

-   Got this message:

INIT_REPORT Init Duration: 124.21 ms Phase: init Status: error Error Type: Runtime.InvalidEntrypoint
INIT_REPORT Init Duration: 6.75 ms Phase: invoke Status: error Error Type: Runtime.InvalidEntrypoint
START RequestId: 6da64f80-0eb1-4cf9-998e-40ceda18fc21 Version: $LATEST
RequestId: 6da64f80-0eb1-4cf9-998e-40ceda18fc21 Error: fork/exec /lambda-entrypoint.sh: exec format error
Runtime.InvalidEntrypoint
END RequestId: 6da64f80-0eb1-4cf9-998e-40ceda18fc21
REPORT RequestId: 6da64f80-0eb1-4cf9-998e-40ceda18fc21 Duration: 26.82 ms Billed Duration: 27 ms Memory Size: 128 MB Max Memory Used: 3 MB

16. Added this line to the Dockerfile

```Dockerfile
RUN chmod +x ${LAMBDA_TASK_ROOT}/main.py
```

17. Tried running Lambda again
    1.  Rebuilt the Docker image `docker build -t lambda-mediapipe .`
    2.  Tagged the image `docker tag lambda-mediapipe:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/lambda-mediapipe:latest`
    3.  Pushed the image to the ECR `docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/lambda-mediapipe:latest`
    4.  Updated the Lambda function
        1.  AWS Console
        2.  Lambda
        3.  RoMA-T_MediaPipe-Function
        4.  Image
        5.  Deploy new image
        6.  Selected latest image
    5.  Still got the same error
18. Tried running it locally
    1.  Saved the test event as event.json
    2.  Ran the docker image in one terminal
		`docker run -p 9000:8080 lambda-mediapipe`
    3.  Ran a curl command in another terminal
`curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d @event.json`
	1.  Got "Syntax error in module 'main': EOL while scanning string literal (main.py, line 62)"
	2.  Apparently syntax like this breaks the Lambda function, but my syntax extension automatically formats it like this:
	
```python
out_path = f"{
	output_dir_path}/analysed_{os.path.basename(video_file_path)}"
```

19. Tried running the Lambda function again
	1.  Turned off format on save in VS Code
	2.  Rebuilt, tagged, and pushed the image
	3.  Ran it locally again
	4.  Got this message: "{"errorMessage": `Unable to import module 'main': libGL.so.1: cannot open shared object file: No such file or directory", "errorType": "Runtime.ImportModuleError", "stackTrace": []}%`
	5.  Apparently, the libGL.so.1 shared library is missing, which is required by OpenCV
20. Added this line to the Dockerfile: `RUN yum install -y mesa-libGL`
    1.  Rebuilt, tagged, and pushed the image
    2.  Ran it locally again
    3.  Got this message: `{"errorMessage": "Unable to import module 'main': No module named 'lib'", "errorType": "Runtime.ImportModuleError", "stackTrace": []}%`
    4.  ~~I had pose.py in the same directory as main.py, but needed it in a folder called "lib" like in the other versions of this project~~ I needed to change the pose.py import in main.py from `import lib.pose as pose` to `import pose`
21. Moved pose.py to a folder called "lib"
	1.  Rebuilt, tagged, and pushed the image
	2.  Ran it locally again
	3.  Got this message: `{"errorMessage": "'type' object is not subscriptable", "errorType": "TypeError", "stackTrace": ["  File \"/var/lang/lib/python3.8/imp.py\", line 234, in load_module\n    return load_source(name, filename, file)\n", "  File \"/var/lang/lib/python3.8/imp.py\", line 171, in load_source\n    module = _load(spec)\n", "  File \"<frozen importlib._bootstrap>\", line 702, in _load\n", "  File \"<frozen importlib._bootstrap>\", line 671, in _load_unlocked\n", "  File \"<frozen importlib._bootstrap_external>\", line 843, in exec_module\n", "  File \"<frozen importlib._bootstrap>\", line 219, in _call_with_frames_removed\n", "  File \"/var/task/main.py\", line 9, in <module>\n    import pose\n", "  File \"/var/task/pose.py\", line 6, in <module>\n    class PoseDetection:\n", "  File \"/var/task/pose.py\", line 159, in PoseDetection\n    def draw_angle(self, landmark_names: tuple[str, str, str]):\n"]}%`
	4.  The error was due to Python 3.8 not supporting the `tuple[str, str, str]` syntax. I added `from typing import Tuple` to the top of pose.py and changed the line to `def draw_angle(self, landmark_names: Tuple[str, str, str]):`
22. Updated Tuple import in pose.py
	1.  Rebuilt, tagged, and pushed the image
	2.  Ran it locally again
	3.  Got this message: `{"errorMessage": "'type' object is not subscriptable", "errorType": "TypeError", "stackTrace": ["  File \"/var/lang/lib/python3.8/imp.py\", line 234, in load_module\n    return load_source(name, filename, file)\n", "  File \"/var/lang/lib/python3.8/imp.py\", line 171, in load_source\n    module = _load(spec)\n", "  File \"<frozen importlib._bootstrap>\", line 702, in _load\n", "  File \"<frozen importlib._bootstrap>\", line 671, in _load_unlocked\n", "  File \"<frozen importlib._bootstrap_external>\", line 843, in exec_module\n", "  File \"<frozen importlib._bootstrap>\", line 219, in _call_with_frames_removed\n", "  File \"/var/task/main.py\", line 93, in <module>\n    def estimate_pose(frame, angles: List[str], est_angles: dict[str, list[float]]):\n"]}%`
	4.  The error was due to Python 3.8 not supporting the `dict[str, list[float]]` syntax. I added `from typing import Lise, Dict` to the top of main.py and pose.py and changed the related lines
23. Updated List and Dict imports in main.py and pose.py
	1.  Rebuilt, tagged, and pushed the image
	2.  Ran it in the AWS Lambda console
	3.  Still got the entrypoint error
24. Added permissions to both main.py and pose.py
```Dockerfile
RUN chmod 644 ${LAMBDA_TASK_ROOT}/main.py
RUN chmod 644 ${LAMBDA_TASK_ROOT}/lib/pose.py
```
25. Tried switching the Architecture to ARM64
	1.  Redeployed the image with the new architecture
	2.  Tested it in the AWS Lambda console
	3.  Got this: `"errorMessage": "2024-06-18T21:06:12.604Z d4069a4b-b67a-46f2-b093-8917ff217c8a Task timed out after 3.01 seconds"`
26. Changed Lambda function timeout to max (15 minutes)
	1.  Tested it in the AWS Lambda console
	2.  Got this: `Matplotlib created a temporary cache directory at /tmp/matplotlib-0wrpqo48 because the default path (/home/sbx_user1051/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing`
27. Updated the Dockerfile to set the MPLCONFIGDIR environment variable
```Dockerfile
ENV MPLCONFIGDIR=/tmp
```
	1.  Tested it in the AWS Lambda console
	2.  Got this: `Runtime exited with error: signal: killed`
	3.  This typically occurs when the Lambda function exceeds its memory allocation or runs out of available memory
28. Updated the Lambda function's memory to max (10240 MB)
	1.  Tested it in the AWS Lambda console
	2.  SUCCESS!!!

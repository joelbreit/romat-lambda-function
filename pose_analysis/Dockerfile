# Use the Amazon Linux 2 base image for Python 3.9
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies
RUN yum install -y python3-devel gcc mesa-libGL

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir opencv-python-headless mediapipe numpy boto3

# Copy the application code
COPY main.py ${LAMBDA_TASK_ROOT}
COPY pose_detection.py ${LAMBDA_TASK_ROOT}

# Set permissions
RUN chmod 755 ${LAMBDA_TASK_ROOT}/main.py
RUN chmod 755 ${LAMBDA_TASK_ROOT}/pose_detection.py

# Set environment variables (if needed)
ENV MPLCONFIGDIR=/tmp

# Set the entry point for the Lambda function
CMD ["main.lambda_handler"]
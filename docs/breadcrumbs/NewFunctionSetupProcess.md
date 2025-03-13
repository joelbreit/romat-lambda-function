# New Function Setup Process

1. Updated both files
2. Updated Dockerfile similar to previous one but with Python 3.12
3. Had problems with yum command
   1. Lambda Python 3.12 apparently replaced yum with dnf
4. Changed Python version to 3.9. This worked for Docker and for running locally, so I am using it.
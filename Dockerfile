# Step 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . .

# Step 4: Update pip to the latest version
RUN python -m pip install --upgrade pip

# Step 5: Install any needed dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Define environment variables
ENV PATH_TO_DATA="https://drive.google.com/uc?id=1jXQIFh6y3xo52byug_UcqBdtZgjOMn_D"

# Step 7: Specify the command to run multiple Python scripts
CMD python cls_agglomerative.py && \
    python cls_DBCSAN.py && \
    python cls_kmean.py

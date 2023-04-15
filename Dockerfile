# Use the official Python 3.8 image as the base image
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Install system dependencies for CARLA, OpenCV, and Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    lsb-release \
    software-properties-common \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libjpeg62-turbo \
    zlib1g \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip

# Install CARLA API for Python
RUN pip install carla==0.9.13

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for Jupyter Lab
EXPOSE 8888

# Set the entry point to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]

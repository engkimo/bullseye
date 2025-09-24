#!/bin/bash
set -e

# Update system
apt-get update
apt-get install -y build-essential git curl awscli

# Install NVIDIA drivers and CUDA
if ! nvidia-smi &> /dev/null; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get -y install cuda-toolkit-12-3
fi

# Set environment variables
echo "export S3_BUCKET=${s3_bucket}" >> /etc/environment
echo "export PROJECT_NAME=${project_name}" >> /etc/environment
echo "export CUDA_HOME=/usr/local/cuda" >> /etc/environment
echo "export PATH=/usr/local/cuda/bin:\$PATH" >> /etc/environment
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> /etc/environment

# Create working directory
mkdir -p /opt/docja
chown ubuntu:ubuntu /opt/docja
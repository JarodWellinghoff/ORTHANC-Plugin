# orthanc/Dockerfile.dev
FROM jodogne/orthanc-python:1.12.9

USER root

# Install Python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip build-essential

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages -r /tmp/requirements.txt

# Copy SSL certificates
COPY ssl/ /etc/ssl/certs/

# Create python directory (code will be mounted)
RUN mkdir /src

# Set working directory
WORKDIR /src
FROM jodogne/orthanc-python:1.12.9

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages -r /tmp/requirements.txt

COPY ssl/ /etc/ssl/certs/

RUN mkdir -p /src/python
WORKDIR /src
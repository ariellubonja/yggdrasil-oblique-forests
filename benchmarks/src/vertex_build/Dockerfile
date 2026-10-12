# Dockerfile for VertexAI
FROM python:3.12
WORKDIR /root

RUN apt-get update && apt-get -y install sudo

# Install YDF from Pip
RUN python3 -m pip install --no-cache-dir ydf

RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY train.py /root/train.py

ENTRYPOINT ["python3", "train.py"]

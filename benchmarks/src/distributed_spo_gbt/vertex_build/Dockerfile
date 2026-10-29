# Vertex AI training image: the fork's ydf wheel (built by wheel/Dockerfile) + train.py.
# Wheel tag via cloudbuild.yaml (_WHEEL_TAG); default = the column-replication wheel.
ARG WHEEL_TAG=34d64d3d
FROM us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/ydf-wheel:${WHEEL_TAG} AS wheel
FROM python:3.12
WORKDIR /root
COPY --from=wheel /wheel/ydf-*.whl /tmp/
RUN python3 -m pip install --no-cache-dir /tmp/ydf-*.whl google-cloud-storage pandas gcsfs
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg
COPY train.py gen_trunk.py train_cells.py /root/
ENTRYPOINT ["python3", "train.py"]

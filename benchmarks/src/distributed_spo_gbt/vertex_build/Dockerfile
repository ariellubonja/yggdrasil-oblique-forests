# Vertex AI training image: the fork's ydf wheel (built by wheel/Dockerfile) + train.py.
FROM us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo/ydf-wheel:cb81b53d AS wheel
FROM python:3.12
WORKDIR /root
COPY --from=wheel /wheel/ydf-*.whl /tmp/
RUN python3 -m pip install --no-cache-dir /tmp/ydf-*.whl google-cloud-storage pandas gcsfs
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg
COPY train.py gen_trunk.py /root/
ENTRYPOINT ["python3", "train.py"]

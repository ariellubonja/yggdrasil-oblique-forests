# Builds the ydf wheel of the public fork (distributed SPO-GBT) from source.
# Same recipe as upstream tools/{build_test_linux,package_linux}.sh, pinned to YDF_COMMIT.
FROM quay.io/pypa/manylinux_2_28_x86_64
ARG YDF_REPO=https://github.com/ariellubonja/yggdrasil-decision-forests
ARG YDF_COMMIT=47fe64fa4a6fc8a4b0a0125a169f0df3f9a0c274
ARG BAZEL_JOBS=24
ADD https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64 /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel
ENV CC=gcc
RUN mkdir /src && cd /src && git init -q && git fetch -q --depth 1 $YDF_REPO $YDF_COMMIT \
    && git checkout -q FETCH_HEAD && git log -1 --oneline
WORKDIR /src/yggdrasil_decision_forests/port/python
RUN python3.12 -m venv /venv && . /venv/bin/activate \
    && pip install -q -r requirements.txt build auditwheel==6.0.0 \
    && bazel build --jobs=$BAZEL_JOBS --config=linux_cpp17 --features=-fully_static_link \
       --copt=-DYDF_USE_DYNAMIC_DISPATCH -- //ydf/...:all \
    && python tools/collect_pip_files.py \
    && (cd tmp_package && python -m build --wheel) \
    && auditwheel repair --plat manylinux_2_28_x86_64 -w /wheel tmp_package/dist/ydf-*-linux_x86_64.whl \
    && ls -l /wheel

#!/usr/bin/env bash
# Wheels + training images for the in-node ablation with the fixes backported (branches abl-*-fix):
# base = 0e37d40d, vqsort = 1a1934fa, innode = 4df36453 (+ parstd = innode built with -DEXACT_STD_SORT=1).
set -uo pipefail
export PATH=$HOME/google-cloud-sdk/bin:$PATH
R=us-east1-docker.pkg.dev/ydf-distributed-spo-gbt/ydf-repo
B=$HOME/abl_build; rm -rf $B; cp -r $HOME/fullfix_vertex/vertex_build $B; cd $B
sed -i 's#^ARG BAZEL_JOBS=24#ARG BAZEL_JOBS=24\nARG BAZEL_EXTRA=#; s#--copt=-DYDF_USE_DYNAMIC_DISPATCH#--copt=-DYDF_USE_DYNAMIC_DISPATCH $BAZEL_EXTRA#' wheel/Dockerfile
sed -i "s#'YDF_COMMIT=\${_YDF_COMMIT}'#'YDF_COMMIT=\${_YDF_COMMIT}', '--build-arg', 'BAZEL_EXTRA=\${_BAZEL_EXTRA}'#; s#^  _IMAGE: .*ydf-wheel:dev'#&\n  _BAZEL_EXTRA: ''#" wheel/cloudbuild.yaml
grep -n BAZEL_EXTRA wheel/Dockerfile wheel/cloudbuild.yaml
build() {  # name commit bazel_extra
  gcloud builds submit wheel --config wheel/cloudbuild.yaml \
    --substitutions "_YDF_COMMIT=$2,_IMAGE=$R/ydf-wheel:abl-$1,_BAZEL_EXTRA=$3" > wheel_$1.log 2>&1; echo "rc=$?" >> wheel_$1.log
  gcloud builds submit . --config cloudbuild.yaml --substitutions _WHEEL_TAG=abl-$1,_IMAGE=$R/train-ydf:ablfix-$1 \
    > train_$1.log 2>&1; echo "rc=$?" >> train_$1.log
}
build base   2e4f45ab6c2efb0c63e318a05ffc0f281c64312e "" &
build vqsort ae2fb273073e3986842f57c5c6f110b09271e672 "" &
build innode b74385178c6c79a6999e96a4e09eee2e242282c3 "" &
build parstd b74385178c6c79a6999e96a4e09eee2e242282c3 "--cxxopt=-DEXACT_STD_SORT=1" &
wait
grep -H "^rc=" wheel_*.log train_*.log

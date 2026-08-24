"""TensorFlow project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps(prefix = ""):
    http_archive(
        # The name should match TF's name for farmhash lib.
        name = "farmhash_archive",
        build_file = prefix + "//third_party/farmhash:farmhash.BUILD",
        # Pinned to a commit (master.zip's checksum drifts as master moves).
        strip_prefix = "farmhash-0d859a811870d10f53a594927d0d0b97573ad06d",
        urls = ["https://github.com/google/farmhash/archive/0d859a811870d10f53a594927d0d0b97573ad06d.tar.gz"],
        sha256 = "18392cf0736e1d62ecbb8d695c31496b6507859e8c75541d7ad0ba092dc52115",
    )

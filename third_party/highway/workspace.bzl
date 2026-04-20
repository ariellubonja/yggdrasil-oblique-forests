"""Google Highway SIMD library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "com_google_highway",
        urls = ["https://github.com/google/highway/archive/refs/tags/1.3.0.tar.gz"],
        strip_prefix = "highway-1.3.0",
        sha256 = "07b3c1ba2c1096878a85a31a5b9b3757427af963b1141ca904db2f9f4afe0bc2",
    )

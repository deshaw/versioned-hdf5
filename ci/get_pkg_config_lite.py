"""Download and extract pkg-config-lite on Windows."""

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import requests

URL_TEMPLATE = (
    "https://downloads.sourceforge.net/project/pkgconfiglite/"
    "{version}/pkg-config-lite-{version}_bin-win32.zip"
)
PKG_CONFIG_LITE_VERSION = "0.28-1"
# Pinned for the version above; SourceForge mirrors serve the same archive.
EXPECTED_SHA256 = "2038c49d23b5ca19e2218ca89f06df18fe6d870b4c6b54c0498548ef88771f6f"
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_TIMEOUT_SECONDS = 60


def download_pkg_config_lite() -> bytes:
    url = URL_TEMPLATE.format(version=PKG_CONFIG_LITE_VERSION)
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
        if response.ok:
            break
        print(
            f"Download of {url} failed (attempt {attempt}/{DOWNLOAD_ATTEMPTS})",
            file=sys.stderr,
        )
    else:
        raise RuntimeError(
            f"Cannot download pkg-config-lite from {url} "
            f"after {DOWNLOAD_ATTEMPTS} attempts"
        )

    actual_sha256 = hashlib.sha256(response.content).hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"SHA256 mismatch for {url}: got {actual_sha256}, "
            f"expected {EXPECTED_SHA256}"
        )
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract pkg-config-lite."
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="directory to extract pkg-config-lite into",
    )
    args = parser.parse_args()

    archive = download_pkg_config_lite()
    print(
        f"Extracting pkg-config-lite {PKG_CONFIG_LITE_VERSION} into {args.destination}"
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as archive_file:
        archive_file.extractall(args.destination)


if __name__ == "__main__":
    main()

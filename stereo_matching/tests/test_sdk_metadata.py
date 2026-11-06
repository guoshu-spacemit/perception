#!/usr/bin/env python3
"""Validate the algorithm attribution and current SDK deployment metadata."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    upstream = json.loads((root / "UPSTREAM.json").read_text())
    contract = json.loads((root / "RUNTIME_CONTRACT.json").read_text())
    algorithm = upstream["algorithm"]
    assert algorithm["repository"] == "https://github.com/TomTomTommi/LiteAnyStereo"
    assert algorithm["license"]["spdx_identifier"] == "MIT"
    license_text = (root / algorithm["license"]["local_file"]).read_text()
    assert algorithm["license"]["copyright"] in license_text
    assert "Permission is hereby granted" in license_text
    assert contract["public_api_header"] == "include/stereo_matching/pipeline.h"
    header = root / contract["public_api_header"]
    assert hashlib.sha256(header.read_bytes()).hexdigest() == contract["public_api_sha256"]
    assert contract["model"]["external_only"]
    assert contract["runtime"]["external_only"]
    assert contract["runtime"]["artifact_manifest"] == "SHA256SUMS"
    assert "source_revision" not in contract
    assert "not a byte-for-byte copy" in upstream["sdk_adaptation"]["relationship"]
    print("PASS upstream algorithm/license, current public API and external deployment metadata")


if __name__ == "__main__":
    main()

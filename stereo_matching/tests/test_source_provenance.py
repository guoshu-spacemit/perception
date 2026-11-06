#!/usr/bin/env python3
"""Check current source bytes without overwriting the original input hashes."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "SOURCE_PROVENANCE.json").read_text())
    modifications = provenance.get("local_modifications", {})
    assert set(modifications) <= set(provenance["files"]), "unknown modified source"
    expected = {"include/las2_usb_stereo.h": provenance["public_header_sha256"]}
    for relative, record in provenance["files"].items():
        assert len(record["sha256"]) == 64, f"missing original hash: {relative}"
        expected[relative] = modifications.get(relative, {}).get(
            "current_sha256", record["sha256"])
    for relative, digest in expected.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == digest, f"current source hash mismatch: {relative}"
    print(f"PASS {len(expected)} source hashes, {len(modifications)} local change record; "
          "original migration hashes retained")


if __name__ == "__main__":
    main()

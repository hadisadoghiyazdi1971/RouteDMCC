#!/usr/bin/env python3
from pathlib import Path
import hashlib, sys
ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "REFERENCE_ARTIFACT_SHA256.txt"
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()
ok=True
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    expected, rel = line.split("  ", 1)
    p = ROOT / rel
    if not p.exists():
        print(f"MISSING  {rel}")
        ok=False
        continue
    actual=sha256(p)
    if actual != expected:
        print(f"FAIL     {rel}\n  expected {expected}\n  actual   {actual}")
        ok=False
    else:
        print(f"OK       {rel}")
sys.exit(0 if ok else 1)

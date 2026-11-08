# Verify that all four downloaded packages were extracted into one folder.
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "FILE_MANIFEST_SHA256.json").read_text(encoding="utf-8"))
missing, modified = [], []
for name, expected in manifest["sha256"].items():
    p = root / name
    if not p.exists():
        missing.append(name)
    elif hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        modified.append(name)
print(f"Expected files: {len(manifest['sha256'])}")
print(f"Missing: {len(missing)}; modified: {len(modified)}")
for item in missing:
    print("MISSING", item)
for item in modified:
    print("MODIFIED", item)
if not missing and not modified:
    print("OK: complete project, all file checksums match")
else:
    raise SystemExit(1)

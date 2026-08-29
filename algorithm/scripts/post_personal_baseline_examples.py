"""Stdlib HTTP client: POST the personal-baseline example bundle to a server.

No third-party dependencies (uses urllib). Point it at a running SheWrist API
and it will submit each mock day, poll the async job, fetch the session result,
and print how the ``personal_baseline`` block evolves.

Start the server first (from the project root):
    PYTHONPATH=src .venv/bin/python -m uvicorn shewrist.api:app --port 8000

Then run:
    .venv/bin/python scripts/post_personal_baseline_examples.py
    # or against another host:
    SHEWRIST_BASE_URL=http://127.0.0.1:8000 .venv/bin/python scripts/post_personal_baseline_examples.py
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE_URL = os.environ.get("SHEWRIST_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
BUNDLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "personal_baseline"


def _encode_multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----shewrist{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += value.encode("utf-8") + b"\r\n"
    for name, path in files.items():
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode()
        body += f"Content-Type: {content_type}\r\n\r\n".encode()
        body += path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _request(method: str, url: str, *, data: bytes | None = None, content_type: str | None = None) -> dict:
    request = urllib.request.Request(url, data=data, method=method)
    if content_type:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        raise SystemExit(f"HTTP {exc.code} on {method} {url}: {payload}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach {url}: {exc}. Is the server running?") from exc


def submit_day(entry: dict) -> dict:
    metadata_path = BUNDLE_DIR / entry["metadata"]
    fields = {"metadata": metadata_path.read_text(encoding="utf-8")}
    files = {
        "data_file": BUNDLE_DIR / entry["data_file"],
        "mechanical_file": BUNDLE_DIR / entry["mechanical_file"],
    }
    body, content_type = _encode_multipart(fields, files)
    job = _request("POST", f"{BASE_URL}/api/v1/analysis-jobs", data=body, content_type=content_type)
    job_id = job["job_id"]
    for _ in range(120):
        status = _request("GET", f"{BASE_URL}/api/v1/analysis-jobs/{job_id}")
        if status["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.25)
    if status["status"] != "succeeded":
        raise SystemExit(f"Job {job_id} ended as {status['status']}: {status.get('error')}")
    return _request("GET", f"{BASE_URL}/api/v1/sessions/{entry['session_id']}")


def main() -> None:
    order = json.loads((BUNDLE_DIR / "post_order.json").read_text(encoding="utf-8"))
    print(f"POSTing {len(order['days'])} mock days to {BASE_URL} ...\n")
    print(f"{'day':>3} {'status':>12} {'sessions':>8} {'assoc':>14} {'r':>6} {'tolerance':>10} {'#sugg':>6}")
    last = None
    for entry in order["days"]:
        result = submit_day(entry)
        pb = result["personal_baseline"]
        assoc = pb["symptom_association"]
        tol = pb["exposure_tolerance"].get("tolerance_exposure")
        r = assoc.get("pearson_r")
        print(
            f"{entry['day']:>3} {pb['status']:>12} {int(pb['session_count']):>8} "
            f"{assoc['status']:>14} {('%.2f' % r) if r is not None else '  -':>6} "
            f"{('%.1f' % tol) if tol is not None else '   -':>10} {len(pb['suggestions']):>6}"
        )
        last = result
    print("\nFinal personal_baseline block:")
    print(json.dumps(last["personal_baseline"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

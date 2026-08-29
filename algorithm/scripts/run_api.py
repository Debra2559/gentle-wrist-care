#!/usr/bin/env python3
"""Run the SheWrist Backend API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        raise SystemExit("The file-backed v1 backend requires --workers 1")
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install API dependencies with: python3 -m pip install -r requirements-api.txt") from exc
    from shewrist.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

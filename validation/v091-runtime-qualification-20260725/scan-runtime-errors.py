#!/usr/bin/env python3
"""Scan runtime logs while excluding semantic-preflight symbol listings."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


NM_SYMBOL = re.compile(r"^\s+[A-Za-z?]\s+\S+")
ERROR = re.compile(
    r"((?:cuda|tensorrt|myelin)[^\n]*(?:\berror\b|\bfailed\b|illegal|"
    r"out of memory|already loaded binary graph)|"
    r"\[trt\][^\n]*(?:\[e\]|\berror\b|\bfail)|"
    r"illegal memory access|segmentation fault|core dumped|terminate called|"
    r"worker[^\n]*(?:\bexit(?:ed)?\b|\bcrash))",
    re.IGNORECASE,
)


def main() -> int:
    hits = []
    ignored_nm_lines = 0
    for name in sys.argv[1:]:
        for number, line in enumerate(
            Path(name).read_text(errors="replace").splitlines(), 1
        ):
            if NM_SYMBOL.match(line):
                ignored_nm_lines += 1
                continue
            if ERROR.search(line):
                hits.append({"file": name, "line": number, "text": line[:500]})
    result = {
        "hits": hits,
        "ignored_nm_symbol_lines": ignored_nm_lines,
        "passed": not hits,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

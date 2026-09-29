#!/usr/bin/env python3
"""Aggregate uvbench UVAtlas timing traces."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    args = parser.parse_args()

    line_re = re.compile(r"\[uvatlas\]\s+(\S.*?)\s+(\d+\.\d+) ms")
    totals: dict[str, list[float]] = defaultdict(list)
    result = ""
    with open(args.log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = line_re.search(line)
            if match:
                totals[match.group(1).strip()].append(float(match.group(2)))
            if line.startswith("RESULT"):
                result = line.strip()

    print(result)
    for label, values in sorted(totals.items(), key=lambda kv: -sum(kv[1])):
        print(f"{sum(values):12.1f} ms  n={len(values):6d}  avg={sum(values)/len(values):9.3f} ms  {label}")


if __name__ == "__main__":
    main()

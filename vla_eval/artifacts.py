from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from .statistics import paired_comparisons, summarize


def metadata(config):
    versions = {}
    for package in ("numpy", "torch", "transformers", "gr00t", "isaac-gr00t"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "platform": platform.platform(),
              "versions": versions, "git_revision": revision, "config": config,
              "evidence_type": "cpu_fixture_only" if config.get("backend", "").startswith(
                  "vla_eval.toy:") else "user_backend_requires_validation",
              "overlap": "sequential timings with analytical reach overlap; no concurrency claim"}
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["cuda_version"] = torch.version.cuda
            result["gpu"] = torch.cuda.get_device_name(0)
    return result


class Artifacts:
    def __init__(self, output, config):
        self.output = Path(output)
        # New directory per run prevents accidental mixing or overwriting results.
        self.output.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.config = config
        (self.output / "manifest.json").write_text(
            json.dumps(metadata(config), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        self.stream = (self.output / "trials.jsonl").open("w", encoding="utf-8")

    def record_backend(self, heavy, small):
        path = self.output / "manifest.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        runtime = metadata(self.config)
        original.update({k: v for k, v in runtime.items() if k != "created_utc"})
        original["heavy_name"] = getattr(heavy, "name", type(heavy).__name__)
        original["small_name"] = getattr(small, "name", type(small).__name__) if small else None
        if hasattr(heavy, "engine") and hasattr(heavy.engine, "metadata"):
            original["model_metadata"] = heavy.engine.metadata
        elif hasattr(heavy, "metadata"):
            original["model_metadata"] = heavy.metadata
        path.write_text(json.dumps(original, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def emit(self, row):
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        self.rows.append(row)

    def finish(self):
        self.stream.close()
        groups = summarize(self.rows)
        comparison = paired_comparisons(self.rows)
        for filename, value in (("summary.json", groups), ("paired_comparisons.json", comparison)):
            (self.output / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                                         allow_nan=False), encoding="utf-8")
        if self.config.get("horizon_analysis", False):
            from .horizon_statistics import horizon_comparisons
            (self.output / "horizon_comparisons.json").write_text(
                json.dumps(horizon_comparisons(self.rows), indent=2, ensure_ascii=False,
                           allow_nan=False), encoding="utf-8")
        flat = [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in self.rows]
        if flat:
            with (self.output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in flat))))
                writer.writeheader()
                writer.writerows(flat)
        return groups

    def fail(self, error):
        self.stream.close()
        (self.output / "FAILED.txt").write_text(str(error), encoding="utf-8")

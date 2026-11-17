"""Download explicit LIBERO episode files and prepare RGB/state/action diagnostics.

Downloads no model weights or complete dataset. Public source episodes retain
their train split; this output is not a held-out accuracy benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np


REPO = "physical-intelligence/libero"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--max-samples-per-episode", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-download-mib", type=int, default=64)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--output", default="data/libero_subset.npz")
    args = parser.parse_args()
    if (not args.episodes or min(args.episodes) < 0 or len(set(args.episodes)) != len(args.episodes)
            or min(args.max_samples_per_episode, args.horizon, args.max_download_mib) < 1):
        raise ValueError("Unique nonnegative episodes and positive size/length limits are required")
    output = Path(args.output)
    provenance_path = output.with_suffix(".source.json")
    if output.suffix != ".npz" or output.exists() or provenance_path.exists():
        raise ValueError("Output must be a new .npz path")
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
    import pyarrow.parquet as pq
    from PIL import Image

    revision = HfApi(token=False).dataset_info(REPO, revision=args.revision).sha
    downloaded_bytes = 0
    sources = []

    def fetch(filename):
        nonlocal downloaded_bytes
        url = hf_hub_url(REPO, filename, repo_type="dataset", revision=revision)
        size = get_hf_file_metadata(url, token=False).size
        if size is None or downloaded_bytes + size > args.max_download_mib * 1024**2:
            raise ValueError("Requested subset exceeds the download size limit or has unknown size")
        path = Path(hf_hub_download(REPO, filename, repo_type="dataset", revision=revision,
                                    cache_dir=args.cache_dir, token=False))
        downloaded_bytes += size
        sources.append({"filename": filename, "bytes": size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return path

    info = json.loads(fetch("meta/info.json").read_text(encoding="utf-8"))
    tasks = {int(x["task_index"]): x["task"] for line in fetch("meta/tasks.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip() for x in [json.loads(line)]}
    if info.get("total_videos", 0) != 0:
        raise ValueError("This importer requires embedded images, not external video files")
    payload = {key: [] for key in ("episode_id", "split", "state", "actions", "action_mask",
                                  "instruction", "image__image", "image__wrist_image",
                                  "source_frame_index")}

    def decode(value):
        if not isinstance(value, dict) or value.get("bytes") is None:
            raise ValueError("Expected an embedded image byte payload")
        with Image.open(io.BytesIO(value["bytes"])) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    for episode in args.episodes:
        if episode >= info["total_episodes"]:
            raise ValueError(f"Episode {episode} is outside this dataset")
        filename = info["data_path"].format(episode_index=episode,
                                             episode_chunk=episode // info["chunks_size"])
        rows = pq.read_table(fetch(filename)).to_pylist()
        if any(int(row["episode_index"]) != episode for row in rows):
            raise ValueError("Episode file contains inconsistent episode indices")
        eligible = len(rows) - args.horizon + 1
        if eligible < 1:
            raise ValueError(f"Episode {episode} is shorter than H={args.horizon}")
        count = min(args.max_samples_per_episode, eligible)
        positions = ([eligible // 2] if count == 1 else
                     np.linspace(0, eligible - 1, count, dtype=int).tolist())
        for position in positions:
            row = rows[position]
            target = np.asarray([r["actions"] for r in rows[position:position + args.horizon]],
                                dtype=np.float32)
            state = np.asarray(row["state"], dtype=np.float32)
            if state.shape != (8,) or target.shape != (args.horizon, 7):
                raise ValueError("Unexpected LIBERO state/action dimensions")
            payload["episode_id"].append(f"libero_{episode:06d}")
            payload["split"].append("train")
            payload["state"].append(state)
            payload["actions"].append(target)
            payload["action_mask"].append(np.ones(args.horizon, dtype=bool))
            payload["instruction"].append(tasks[int(row["task_index"])])
            payload["image__image"].append(decode(row["image"])[None])
            payload["image__wrist_image"].append(decode(row["wrist_image"])[None])
            payload["source_frame_index"].append(int(row["frame_index"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{k: np.asarray(v) for k, v in payload.items()})
    provenance = {"source_repo": REPO, "source_revision": revision, "source_split": "train",
                  "diagnostic_only": True, "checkpoint_training_overlap_possible": True,
                  "episodes": args.episodes, "samples": len(payload["state"]),
                  "horizon": args.horizon, "source_fps_metadata": info["fps"],
                  "selection": "evenly spaced full-target windows per episode",
                  "downloaded_bytes": downloaded_bytes, "files": sources,
                  "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": provenance["samples"],
                      "episodes": args.episodes, "downloaded_mib": round(downloaded_bytes / 1024**2, 2),
                      "output_mib": round(output.stat().st_size / 1024**2, 2),
                      "diagnostic_only": True}, indent=2))


if __name__ == "__main__":
    main()

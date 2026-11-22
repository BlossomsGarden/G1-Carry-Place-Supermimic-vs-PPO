"""Package physical evaluations and training costs without large checkpoint files.

Run on the training server. The resulting archive can be transferred to the
workstation and extracted into docs/metrics/learning_curve/. No rollout NPZs
or checkpoint tensors are included in this small audit bundle.
"""

import argparse
import io
import json
import re
import tarfile
from pathlib import Path


def archive_run(archive, folder, label, slug):
    evaluation = folder / (slug + "_evaluations")
    for path in sorted(evaluation.glob("*")):
        if path.name in ("monitor_state.json", "monitor_pid.json") or (
            path.name.startswith("iter") and path.name.endswith(("_metrics.json", "_eval.json", ".log"))
        ):
            archive.add(path, arcname=f"{label}/{path.name}")
    for suffix in (".log", "_gpu_samples.csv", "_run_metadata.json",
                   "_start_utc.txt", "_end_utc.txt", "_exit_code.txt",
                   "_controlled_stop.json"):
        path = folder / (slug + suffix)
        if path.exists():
            archive.add(path, arcname=f"{label}/{path.name}")
    log = (folder / (slug + ".log")).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\[train_track\].*? -> (/[^\s]+)", log)
    if not match:
        return
    run_directory = Path(match.group(1))
    if not run_directory.is_dir():
        raise FileNotFoundError(run_directory)
    manifest = []
    checkpoints = (p for p in run_directory.glob("model_*.pt")
                   if p.stem.rsplit("_", 1)[-1].isdigit())
    for checkpoint in sorted(checkpoints, key=lambda p: int(p.stem.rsplit("_", 1)[-1])):
        stat = checkpoint.stat()
        manifest.append({"iteration": int(checkpoint.stem.split("_")[-1]),
                         "absolute_path": str(checkpoint), "bytes": stat.st_size,
                         "mtime_unix": stat.st_mtime})
    output = json.dumps(manifest, indent=2).encode("utf-8")
    item = tarfile.TarInfo(f"{label}/checkpoint_manifest.json")
    item.size = len(output)
    archive.addfile(item, io.BytesIO(output))
    for config in sorted((run_directory / "params").glob("*.json")):
        archive.add(config, arcname=f"{label}/params/{config.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--single-run", required=True)
    parser.add_argument("--four-run", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        archive_run(archive, args.runs_root, "single", args.single_run)
        archive_run(archive, args.runs_root, "four", args.four_run)
    print(args.output, args.output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

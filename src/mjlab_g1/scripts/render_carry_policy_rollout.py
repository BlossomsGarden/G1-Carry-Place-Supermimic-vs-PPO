"""Render an independently scored, physically simulated CarryPlace policy rollout.

The input states and contacts come from eval_track.py --rollout-npz, not the SUGAR
reference motion. Rendering only repositions the scene to recorded simulated states;
no kinematic reference frame is substituted for a policy action. By default only
strict successes are accepted; --allow-failure marks unsuccessful trials explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import mujoco
from mujoco.rendering.classic.renderer import Renderer
import numpy as np
from PIL import Image, ImageDraw

from render_carry_reference import build_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rollout-npz", type=Path, required=True)
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--g1-xml", type=Path, required=True)
    p.add_argument("--box-mesh", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contact-sheet", type=Path)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--allow-failure", action="store_true",
                   help="render a failed physical policy rollout with its failure label")
    args = p.parse_args()
    report = json.loads(args.eval_json.read_text(encoding="utf-8"))
    with np.load(args.rollout_npz, allow_pickle=False) as z:
        rollout = {name: z[name].copy() for name in z.files}
    idx = int(rollout["episode_index"])
    if not 0 <= idx < report["episodes"]:
        raise ValueError("rollout episode index is outside evaluation")
    episode = report["per_episode"][idx]
    state = rollout["state"]
    if (str(rollout["checkpoint"]) != report["checkpoint"]
            or int(rollout["seed"]) != report["seed"]
            or bool(rollout["randomized"]) != (report["condition"] == "randomized")
            or int(episode["episode_steps"]) + 1 != len(state)
            or bool(rollout["episode_success"]) != bool(episode["success"])
            or (not args.allow_failure and not episode["success"])):
        raise ValueError("rollout metadata disagrees with its physical evaluation or success requirement")
    if state.ndim != 2 or state.shape[1] != 84 or not np.isfinite(state).all():
        raise ValueError("expected finite [frames, 84] G1 plus box simulator states")
    if args.fps <= 0 or 50 % args.fps:
        raise ValueError("--fps must be a positive divisor of the 50 Hz control rate")
    length = len(state)
    target = rollout["target_pos_w"]
    model = build_model(args.g1_xml, args.box_mesh, target)
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.mean(state[:, :3], axis=0) * [1, 1, 0] + [0, 0, 0.70]
    cam.distance = 3.45
    cam.azimuth = 140
    cam.elevation = -15
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}",
         "-r", str(args.fps), "-i", "-", "-an", "-vcodec", "libx264",
         "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags",
         "+faststart", str(args.output)], stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    stride = 50 // args.fps
    key_indices = [0, min(length - 1, 150), min(length - 1, 210),
                   min(length - 1, 350), min(length - 1, 475), length - 1]
    keyframes = {}
    try:
        with Renderer(model, height=args.height, width=args.width) as renderer:
            for k in range(0, length):
                if k % stride and k not in key_indices:
                    continue
                data.qpos[:36] = state[k, :36]
                data.qpos[36:43] = state[k, 71:78]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle([0, 0, args.width, 83], fill=(9, 14, 24))
                verdict = "STRICT SUCCESS" if episode["success"] else "STRICT FAILURE"
                draw.text((10, 7), f"{report['learner'].upper()} POLICY | REAL MUJOCO ROLLOUT | {verdict}",
                          fill="white")
                draw.text((10, 31),
                          f"seed {report['seed']}  episode {idx}  frame {k}/{length-1}  t={k/50:.2f}s",
                          fill="white")
                draw.text((10, 55),
                          f"physical contacts L/R/floor: "
                          f"{int(rollout['left_hand_contact'][k])}/"
                          f"{int(rollout['right_hand_contact'][k])}/"
                          f"{int(rollout['floor_contact'][k])}", fill="white")
                if k in key_indices:
                    keyframes[k] = frame.copy()
                if k % stride == 0:
                    ffmpeg.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        ffmpeg.stdin.close()
    error = ffmpeg.stderr.read().decode("utf-8", "replace")
    if ffmpeg.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {error}")
    if args.contact_sheet:
        sheet = Image.new("RGB", (args.width * len(key_indices), args.height), "#222222")
        for i, k in enumerate(key_indices):
            sheet.paste(keyframes[k], (i * args.width, 0))
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.contact_sheet)
    print(f"Real policy rollout: success={episode['success']}, "
          f"carried={episode['carried_m']:.3f} m, "
          f"target_error={episode['target_error_m']:.3f} m; "
          f"wrote {args.output} ({length} simulated states)")


if __name__ == "__main__":
    main()

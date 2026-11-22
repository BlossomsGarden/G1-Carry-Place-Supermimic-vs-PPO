"""Render the converted CarryPlace reference with the actual MuJoCo G1 and box.

This is a KINEMATIC replay: qpos is prescribed each video frame. Contacts are
NOT evidence of a trained policy or of dynamically feasible robot actions.
The small box is a dynamic collision mesh in the model for later physical tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import mujoco
from mujoco.rendering.classic.renderer import Renderer
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def build_model(g1_xml: Path, box_mesh: Path, target: np.ndarray) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(g1_xml.resolve()))
    spec.add_mesh(name="sugar_small_box", file=str(box_mesh.resolve()))
    body = spec.worldbody.add_body(name="box")
    body.add_freejoint(name="box_freejoint")
    geom = body.add_geom(name="box_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
                         meshname="sugar_small_box", mass=0.5)
    geom.condim = 3
    geom.friction = [1.0, 0.005, 0.0001]
    geom.priority = 1
    geom.rgba = [0.85, 0.55, 0.25, 1]
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=[5, 5, 0.05], rgba=[0.42, 0.45, 0.47, 1])
    floor.friction = [1.0, 0.005, 0.0001]
    spec.worldbody.add_site(name="placement_target", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            pos=[float(target[0]), float(target[1]), 0.04],
                            size=[0.09, 0.09, 0.09], rgba=[0.15, 0.95, 0.3, 0.65])
    model = spec.compile()
    if model.nq != 43 or model.body("box").id != 31:
        raise ValueError(f"Unexpected G1 + box composition: nq={model.nq}")
    return model


def label(image: np.ndarray, frame: int, fps: int, grasp: int,
          release: int, box: np.ndarray, target: np.ndarray, clip: str) -> Image.Image:
    pic = Image.fromarray(image)
    draw = ImageDraw.Draw(pic)
    phase = "APPROACH / GRASP" if frame < grasp else "CARRY" if frame < release else "PLACE / RELEASE"
    draw.rectangle([0, 0, pic.width, 83], fill=(9, 14, 24))
    lines = ["KINEMATIC REFERENCE - NOT A TRAINED POLICY / NO ACTUATION",
             f"SUGAR CarryBox {clip}  |  frame {frame:03d}  t={frame / 50:.2f}s  |  {phase}",
             f"box xy=({box[0]:+.2f}, {box[1]:+.2f}) m   goal=({target[0]:+.2f}, {target[1]:+.2f}) m"
             f"   source contact label={grasp <= frame < release}"]
    for i, line in enumerate(lines):
        draw.text((10, 7 + i * 24), line, fill=(255, 255, 255))
    return pic


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--motion", required=True, type=Path)
    p.add_argument("--g1-xml", required=True, type=Path)
    p.add_argument("--box-mesh", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--contact-sheet", type=Path)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args()
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        # mjlab installs imageio-ffmpeg's bundled Linux executable in wan.
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    with np.load(args.motion) as z:
        motion = {name: z[name] for name in z.files}
    length = len(motion["joint_pos"])
    grasp, release = int(motion["grasp_frame"]), int(motion["release_frame"])
    clip = str(motion["source_clip"].item())
    target = motion["target_pos_w"]
    model = build_model(args.g1_xml, args.box_mesh, target)
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.mean(motion["body_pos_w"][:, 0], axis=0) * [1, 1, 0] + [0, 0, 0.70]
    cam.distance = 3.45
    cam.azimuth = 140
    cam.elevation = -15
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}",
         "-r", str(args.fps), "-i", "-", "-an", "-vcodec", "libx264",
         "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags",
         "+faststart", str(args.output)], stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    keyframes = {}
    interest = [max(0, grasp - 20), min(grasp + 20, length - 1),
                (grasp + release) // 2, max(0, release - 20), length - 1]
    stride = max(1, round(50 / args.fps))
    with Renderer(model, height=args.height, width=args.width) as renderer:
        for k in range(length):
            if k % stride and k not in interest:
                continue
            data.qpos[:3] = motion["body_pos_w"][k, 0]
            data.qpos[3:7] = motion["body_quat_w"][k, 0]
            data.qpos[7:36] = motion["joint_pos"][k]
            data.qpos[36:39] = motion["box_pos_w"][k]
            data.qpos[39:43] = motion["box_quat_w"][k]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frame = label(renderer.render(), k, args.fps, grasp, release,
                          motion["box_pos_w"][k], target, clip)
            if k in interest:
                keyframes[k] = frame.copy()
            if k % stride == 0:
                ffmpeg.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    ffmpeg.stdin.close()
    error = ffmpeg.stderr.read().decode("utf-8", "replace")
    if ffmpeg.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {error}")
    if args.contact_sheet:
        sheet = Image.new("RGB", (args.width * len(interest), args.height), "#222222")
        for i, k in enumerate(interest):
            sheet.paste(keyframes[k], (i * args.width, 0))
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.contact_sheet)
    print(f"Rendered {length} reference frames ({length / 50:.2f}s) -> {args.output}")
    print(f"This is prescribed qpos, not a policy or physical success trial; keyframes={interest}")


if __name__ == "__main__":
    main()

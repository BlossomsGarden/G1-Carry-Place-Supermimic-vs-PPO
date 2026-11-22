"""Extract SUGAR's scanned small-box convex collision hull from the official USD.

Preprocessing needs usd-core (pxr), scipy, and numpy. The generated OBJ contains
only collision vertices/faces and is used with a 0.5 kg dynamic MuJoCo body.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--usd", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(args.usd))
    if not stage:
        raise ValueError(f"Failed to open USD {args.usd}")
    meshes = []
    for instance in stage.TraverseAll():
        if instance.IsInstance():
            meshes.extend(prim for prim in Usd.PrimRange(instance.GetPrototype())
                          if prim.GetTypeName() == "Mesh" and prim.HasAPI(UsdPhysics.CollisionAPI))
    if len(meshes) != 1:
        raise ValueError(f"Expected one SUGAR small-box collision mesh, got {len(meshes)}")
    vertices = np.asarray(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get(), dtype=np.float64)
    hull = ConvexHull(vertices)
    selected = np.asarray(hull.vertices)
    index = {int(original): i + 1 for i, original in enumerate(selected)}
    lines = ["# Convex collision hull derived from SUGAR official small_box/obj_aligned.usd",
             "# Original local coordinates and metres preserved"]
    lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices[selected])
    for face, equation in zip(hull.simplices, hull.equations):
        a, b, c = (vertices[i] for i in face)
        if np.dot(np.cross(b - a, c - a), equation[:3]) < 0:
            face = face[[0, 2, 1]]
        lines.append("f " + " ".join(str(index[int(i)]) for i in face))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="ascii")
    report = dict(source_usd_sha256=hashlib.sha256(args.usd.read_bytes()).hexdigest(),
                  collision_obj_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  input_vertices=len(vertices), convex_vertices=len(selected),
                  convex_faces=len(hull.simplices), volume_m3=hull.volume,
                  dimensions_aabb_m=(vertices.max(axis=0) - vertices.min(axis=0)).round(4).tolist(),
                  source="SUGAR official processed descriptions.zip: small_box/obj_aligned.usd")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

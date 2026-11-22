"""Extract only SUGAR's CarryBox inputs, optionally from a partial data.zip.

Use --central-directory only if the ZIP download stopped after the CarryBox
entries: this is the exact central-directory byte range from the same ZIP.
Ordinary complete ZIPs need no extra argument. Each entry is CRC checked.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import zipfile
import zlib

PREFIX = "data/CarryBox/"
SUFFIXES = ("robot_50hz.npz", "obj_motion_global_50hz.pkl", "contact_labels_50hz.npy")
CENTRAL = struct.Struct("<IHHHHHHIIIHHHHHII")


def partial_entries(central_path: Path):
    data = central_path.read_bytes()
    position = 0
    while position < len(data):
        if data[position:position + 4] != b"PK\x01\x02":
            raise ValueError(f"Broken central directory at byte {position}")
        fields = CENTRAL.unpack_from(data, position)
        length, extra, comment = fields[10:13]
        name = data[position + 46:position + 46 + length].decode("utf-8")
        position += 46 + length + extra + comment
        if name.startswith(PREFIX) and name.endswith(SUFFIXES):
            yield name, fields[4], fields[7], fields[8], fields[9], fields[16]


def extract_partial(archive: Path, central_path: Path, destination: Path) -> int:
    count = 0
    with archive.open("rb") as file:
        for name, method, crc, compressed_size, raw_size, offset in partial_entries(central_path):
            # Protect against a damaged archive or an unexpected path.
            components = Path(name[len(PREFIX):]).parts
            if len(components) != 2 or not components[0].startswith("data_"):
                raise ValueError(f"Unsafe or unexpected path: {name}")
            file.seek(offset)
            local = file.read(30)
            if len(local) != 30 or local[:4] != b"PK\x03\x04":
                raise ValueError(f"Missing local ZIP header: {name}")
            name_size, extra_size = struct.unpack_from("<HH", local, 26)
            file.seek(name_size + extra_size, 1)
            contents = file.read(compressed_size)
            if len(contents) != compressed_size:
                raise ValueError(f"Partial archive ends inside {name}")
            if method == 0:
                raw = contents
            elif method == 8:
                raw = zlib.decompress(contents, -15)
            else:
                raise ValueError(f"Unsupported ZIP compression {method}: {name}")
            if len(raw) != raw_size or zlib.crc32(raw) != crc:
                raise ValueError(f"Size/CRC mismatch: {name}")
            out = destination.joinpath(*components)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
            count += 1
    return count


def extract_complete(archive: Path, destination: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive) as file:
        for name in file.namelist():
            if not name.startswith(PREFIX) or not name.endswith(SUFFIXES):
                continue
            components = Path(name[len(PREFIX):]).parts
            if len(components) != 2 or not components[0].startswith("data_"):
                raise ValueError(f"Unsafe or unexpected path: {name}")
            out = destination.joinpath(*components)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(file.read(name))
            count += 1
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--central-directory", type=Path)
    args = p.parse_args()
    if args.central_directory:
        count = extract_partial(args.archive, args.central_directory, args.output)
    else:
        count = extract_complete(args.archive, args.output)
    directories = sorted(args.output.glob("data_*"))
    if count != 3 * len(directories) or not directories:
        raise ValueError(f"Incomplete CarryBox extraction: {count} entries in {len(directories)} clips")
    print(f"Extracted {len(directories)} CarryBox clips ({count} verified files) into {args.output}")


if __name__ == "__main__":
    main()

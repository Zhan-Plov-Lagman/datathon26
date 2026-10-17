"""Repack a local safetensors checkpoint with bounded RAM, without changing weights."""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path


def shard_model(source_dir: Path, output_dir: Path, max_bytes: int = 512 * 2**20) -> dict:
    if max_bytes <= 0:
        raise ValueError("Shard size must be positive")
    source = source_dir / "model.safetensors"
    with source.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size > 100 * 2**20:
            raise ValueError("Invalid safetensors header")
        header = json.loads(handle.read(header_size))
    groups, current, current_size, total_size = [], [], 0, 0
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        start, end = descriptor["data_offsets"]
        size = end - start
        if current and current_size + size > max_bytes:
            groups.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += size
        total_size += size
    if current:
        groups.append(current)
    output_dir.mkdir(parents=True, exist_ok=False)
    weight_map = {}
    # Copy raw tensor bytes, including BF16, without a whole-file writable mapping.
    # Safetensors format: uint64 header length, padded JSON, contiguous tensor data.
    with source.open("rb") as weights:
        for index, names in enumerate(groups, 1):
            filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            shard_header = {"__metadata__": header.get("__metadata__", {"format": "pt"})}
            offset = 0
            for name in names:
                descriptor = header[name]
                start, end = descriptor["data_offsets"]
                shard_header[name] = {**descriptor, "data_offsets": [offset, offset + end - start]}
                offset += end - start
            encoded = json.dumps(shard_header, separators=(",", ":")).encode("utf-8")
            encoded += b" " * (-len(encoded) % 8)
            with (output_dir / filename).open("xb") as target:
                target.write(struct.pack("<Q", len(encoded)))
                target.write(encoded)
                for name in names:
                    start, end = header[name]["data_offsets"]
                    weights.seek(8 + header_size + start)
                    remaining = end - start
                    while remaining:
                        chunk = weights.read(min(remaining, 8 * 2**20))
                        if not chunk:
                            raise ValueError("Truncated checkpoint")
                        target.write(chunk)
                        remaining -= len(chunk)
            weight_map.update({name: filename for name in names})
    for path in source_dir.iterdir():
        if path.is_file() and (path.suffix in (".json", ".txt", ".jinja") or path.name.startswith("LICENSE")):
            if path.name != "model.safetensors.index.json":
                shutil.copy2(path, output_dir / path.name)
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2), encoding="utf-8")
    result = {"shards": len(groups), "tensors": len(weight_map), "total_tensor_bytes": total_size}
    (output_dir / "repack_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", default="models/Qwen3-VL-2B-Instruct-shards")
    args = parser.parse_args()
    print(json.dumps(shard_model(Path(args.source_dir), Path(args.output_dir))))


if __name__ == "__main__":
    main()

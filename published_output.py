"""Publish derived images only after a complete same-filesystem write."""

import os
import struct
import uuid


def complete_v3draw(path):
    """Validate the V3DRAW header and exact payload size without reading TBs."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(43)
        if len(header) != 43 or header[:24].rstrip(b"\0") != b"raw_image_stack_by_hpeng":
            return False
        endian = {ord("L"): "<", ord("B"): ">"}.get(header[24])
        if endian is None:
            return False
        datatype = struct.unpack(f"{endian}h", header[25:27])[0]
        if datatype not in {1, 2, 4}:
            return False
        dimensions = struct.unpack(f"{endian}4I", header[27:43])
        if any(value == 0 for value in dimensions):
            return False
        payload = datatype
        for dimension in dimensions:
            payload *= dimension
        return os.path.getsize(path) == 43 + payload
    except (OSError, IndexError, struct.error):
        return False


def complete_tiff(path):
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return os.path.getsize(path) > 0
    except (OSError, ValueError):
        return False


def nonempty_file(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def publish_derived_file(target, build, validate):
    """Skip a valid result; rebuild an incomplete one without exposing it."""
    if os.path.isfile(target) and validate(target):
        return target
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    stem, extension = os.path.splitext(os.path.basename(target))
    temporary = os.path.join(
        parent, f".{stem}.{uuid.uuid4().hex}.partial{extension}"
    )
    try:
        if build(temporary) is False or not validate(temporary):
            raise RuntimeError(f"Derived output is incomplete: {target}")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

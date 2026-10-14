"""Power-loss recovery checks for derived-image publication."""

import struct
import tempfile
import unittest
from pathlib import Path

from published_output import complete_v3draw, publish_derived_file


def tiny_v3draw(payload=b"\x01\x02"):
    return (
        struct.pack("<24s", b"raw_image_stack_by_hpeng")
        + b"L"
        + struct.pack("<h", 1)
        + struct.pack("<4I", 2, 1, 1, 1)
        + payload
    )


class PublishedOutputTests(unittest.TestCase):
    def test_v3draw_rejects_truncated_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "image.v3draw"
            target.write_bytes(tiny_v3draw())
            self.assertTrue(complete_v3draw(target))
            target.write_bytes(tiny_v3draw(b"\x01"))
            self.assertFalse(complete_v3draw(target))

    def test_failed_build_does_not_publish_partial_result(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "image.v3draw"

            def interrupted(temporary):
                Path(temporary).write_bytes(tiny_v3draw(b"\x01"))
                raise OSError("simulated power loss")

            with self.assertRaises(OSError):
                publish_derived_file(target, interrupted, complete_v3draw)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).glob("*.partial*")), [])

            def complete(temporary):
                Path(temporary).write_bytes(tiny_v3draw())

            self.assertEqual(
                publish_derived_file(target, complete, complete_v3draw),
                target,
            )
            self.assertTrue(complete_v3draw(target))

    def test_legacy_incomplete_output_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "image.v3draw"
            target.write_bytes(tiny_v3draw(b"\x01"))
            calls = []

            def complete(temporary):
                calls.append(temporary)
                Path(temporary).write_bytes(tiny_v3draw())

            publish_derived_file(target, complete, complete_v3draw)
            self.assertEqual(len(calls), 1)
            self.assertEqual(target.read_bytes(), tiny_v3draw())
            publish_derived_file(target, complete, complete_v3draw)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()

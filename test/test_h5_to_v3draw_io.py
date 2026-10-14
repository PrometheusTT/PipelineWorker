"""Keep H5 layout discovery metadata-only and read each Z slab once."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import h5py
    import numpy as np
    import pipeline_stage_h5_to_v3draw as stage
    from published_output import complete_v3draw
except ImportError:
    h5py = None


@unittest.skipUnless(h5py is not None, "H5 processing dependencies unavailable")
class H5ToV3drawIOTests(unittest.TestCase):
    def test_layout_discovery_does_not_read_image_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.pyramid.h5"
            with h5py.File(source, "w") as handle:
                pyramid = handle.create_group("0")
                pyramid.create_group("0").create_dataset(
                    "0", data=np.zeros((4, 2, 2), dtype=np.uint16)
                )
                level = pyramid.create_group("1")
                for index in range(4):
                    level.create_dataset(
                        str(index), data=np.full((4, 2, 2), index, dtype=np.uint16)
                    )

            original_getitem = h5py.Dataset.__getitem__
            reads = []

            def guarded_getitem(dataset, key):
                self.assertNotEqual(key, (), "whole H5 block was read")
                reads.append(key)
                return original_getitem(dataset, key)

            with patch.object(h5py.Dataset, "__getitem__", guarded_getitem):
                shapes, arrangements = stage.get_shape_from_h5file(str(source))
                self.assertEqual(arrangements["1"], (2, 2))
                self.assertEqual(shapes["1"], (4, 4, 4))
                dimensions, _, _ = stage.calculate_dimensions(str(source), "1", 2, 2)
                self.assertEqual(dimensions, (4, 4, 4))
                self.assertEqual(reads, [])

                output = Path(directory) / "sample.v3draw"
                stage.process_h5_to_v3draw(
                    str(source), directory, chunk_size_z=2, output_file=str(output)
                )

            self.assertTrue(complete_v3draw(str(output)))
            self.assertEqual(len(reads), 8)
            self.assertTrue(all(isinstance(key, slice) for key in reads))
            self.assertEqual({(key.start, key.stop) for key in reads}, {(0, 2), (2, 4)})
            pixels = np.fromfile(output, dtype="<u2", offset=43).reshape(4, 4, 4)
            expected_plane = np.array(
                [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]],
                dtype=np.uint16,
            )
            np.testing.assert_array_equal(pixels, np.broadcast_to(expected_plane, pixels.shape))


if __name__ == "__main__":
    unittest.main()

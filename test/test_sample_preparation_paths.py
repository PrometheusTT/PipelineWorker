import tempfile
import unittest
from pathlib import Path

from sample_preparation_paths import resolve_crop_metadata_root


class CropMetadataPathsTests(unittest.TestCase):
    def test_prefers_complete_primary_then_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            primary = Path(temp) / "HNDB_files" / "SamplePreparation"
            legacy = Path(temp) / "SamplePreparation"
            doc = "P00001-T001-R001-S001-B1"
            old = legacy / doc / doc
            old.mkdir(parents=True)
            (old / "cell_with_cellID.csv").write_text("Cell ID\n1\n")
            (old / f"{doc}_initial.apo").write_text("x,y,z\n1,2,3\n")
            self.assertEqual(
                resolve_crop_metadata_root(str(primary), str(legacy), doc, doc),
                str(legacy),
            )
            new = primary / doc / doc
            new.mkdir(parents=True)
            (new / "cell_with_cellID.csv").write_text("Cell ID\n1\n")
            (new / f"{doc}_initial.apo").write_text("x,y,z\n1,2,3\n")
            self.assertEqual(
                resolve_crop_metadata_root(str(primary), str(legacy), doc, doc),
                str(primary),
            )

    def test_requires_both_files_and_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                resolve_crop_metadata_root(temp, temp, "doc", "doc")
            with self.assertRaises(ValueError):
                resolve_crop_metadata_root(temp, temp, "../escape", "doc")


if __name__ == "__main__":
    unittest.main()

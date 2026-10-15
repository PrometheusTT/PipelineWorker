"""Select one complete metadata tree for a document without moving files."""

from pathlib import Path


def resolve_crop_metadata_root(primary: str, legacy: str, ptrsid: str, docid: str) -> str:
    if any("/" in value or "\\" in value or value in {"", ".", ".."} for value in (ptrsid, docid)):
        raise ValueError("Invalid crop document name")
    for root in dict.fromkeys((primary, legacy)):
        folder = Path(root) / ptrsid / docid
        if (folder / "cell_with_cellID.csv").is_file() and (folder / f"{docid}_initial.apo").is_file():
            return str(Path(root))
    raise FileNotFoundError(
        f"Crop metadata missing for {docid} in both primary and legacy SamplePreparation roots"
    )

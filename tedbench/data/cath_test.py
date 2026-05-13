import json
from pathlib import Path
from typing import Callable, Optional

import torch
from rich.progress import track

from minesm.utils.structure.protein_chain import ProteinChain

from .abstract import StructureDataset

_CATH_URL = "https://datashare.mpcdf.mpg.de/s/pjXMpff7GsYTR22/download"


class CATHTestDataset(StructureDataset):
    """CATH 4.4 experimental test set.

    Reads pre-processed PDB files from ``<root>/raw/`` and CATH topology
    annotations from ``<root>/cath_labels.json``.  Class-integer label mapping
    is loaded from ``<root>/cath_mapping.pt``, which is shared with the
    TEDBench training splits.

    If the raw data directory does not exist the dataset is automatically
    downloaded from ``_CATH_URL`` and extracted into *root*.

    On first use the dataset is processed and cached to
    ``<root>/processed_cath.pt``; subsequent instantiations load the cache.

    Args:
        root: Path to the CATH dataset directory (``datasets/cath`` by default).
        split: Unused; kept for API consistency with other dataset classes.
        transform: Optional per-sample transform applied in ``__getitem__``.
    """

    def __init__(
        self,
        root: str = "./datasets/cath",
        split="test",
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.raw_dir = self.root / "raw"

        if not self.processed_path.exists():
            if not self.raw_dir.exists() or not any(self.raw_dir.iterdir()):
                from tedbench.utils.io import download_and_extract
                print(f"CATH raw data not found in {self.root}. Downloading …")
                download_and_extract(_CATH_URL, self.root, archive_name="cath.tar.gz")

        self.cath_mapping = torch.load(
            self.root / "cath_mapping.pt", weights_only=False
        )
        self.get_cath_inv_map()
        if not self.processed_path.exists():
            self.process()

        self.data = torch.load(self.processed_path, weights_only=False)

    @property
    def processed_path(self):
        return self.root / "processed_cath.pt"

    def get_cath_inv_map(self):
        self.cath_inv_map_path = self.root / "cath_inv_map.pt"
        if self.cath_inv_map_path.exists():
            self.cath_inv_map = torch.load(self.cath_inv_map_path, weights_only=False)
            return
        self.cath_inv_map = {
            self.cath_mapping[i].item(): i for i in range(len(self.cath_mapping))
        }
        torch.save(self.cath_inv_map, self.cath_inv_map_path)

    def process(self):
        with open(self.root / "cath_labels.json") as f:
            cath_labels = json.load(f)

        pdb_files = sorted(self.raw_dir.glob("*.pdb"))
        data_all = []

        for pdb_file in track(pdb_files, description="Processing..."):
            # Filename is "{pdb}_{chain}.pdb"; chain name uses dot separator
            stem = pdb_file.stem  # e.g. "1abc_A" or "1abc-a" (lowercase chains)
            chain_id = stem[5:]   # auth chain ID, e.g. "A" or "a"
            name = stem[:4] + "." + chain_id  # e.g. "1abc.A" or "1abc.a"

            cath_code = cath_labels.get(name)

            try:
                protein_chain = ProteinChain.from_pdb(pdb_file, id=name)
            except Exception as e:
                print(f"Skipping {name}: {e}")
                continue

            data = self.process_protein_chain(protein_chain, max_length=float("inf"))
            if data is None:
                continue

            cath_code = sorted(cath_code, key=lambda x: x[1], reverse=True)
            cath_code = cath_code[0][0]
            cath_label = self.cath_inv_map.get(cath_code, None)
            if cath_label is None:
                cath_label = self.cath_inv_map.get(
                    ".".join(cath_code.split(".")[:-1]) + ".x", None
                )
            if cath_label is None:
                continue

            data["label"] = cath_label
            data_all.append(data)

        torch.save(data_all, self.processed_path)

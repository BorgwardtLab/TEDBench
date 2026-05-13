import joblib
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from rich.progress import track
from minesm.utils.structure.protein_chain import ProteinChain
from .abstract import StructureDataset

_AFDB_URL = "https://datashare.mpcdf.mpg.de/s/m4owC3SQbd2r6rk/download"


class AFDBDataset(StructureDataset):
    def __init__(
        self,
        root: "str | Path",
        split: str = "train",
        transform: Optional[Callable] = None,
        n_jobs: int = -1,
    ) -> None:
        self.split = split
        file_idx = {"train": 0, "val": 1}
        self.file_idx = file_idx[split]
        self.n_jobs = n_jobs
        self.root = Path(root)
        self.transform = transform
        self.pdb_dir = self.root / "raw"
        self.processed_dir = self.root / "processed_files"
        if not self.processed_paths[self.file_idx].exists():
            self.process()
        self.data = torch.load(self.processed_paths[self.file_idx], weights_only=False)

    @property
    def processed_paths(self):
        return [self.processed_dir / "train_st.pt", self.processed_dir / "val_st.pt"]

    def process_func(self, pdb_file: Path) -> Optional[dict]:
        # Process a single PDB file and return a Dictionary
        protein_chain = ProteinChain.from_pdb(pdb_file)
        return self.process_protein_chain(protein_chain)

    def process(self) -> None:
        # Read data into huge `Data` list.
        pdb_files = sorted(self.pdb_dir.glob("*.pdb"))

        data_list = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self.process_func)(pdb_file)
            for pdb_file in track(pdb_files, description="Processing...")
        )

        data_list = list(filter(lambda data: data is not None, data_list))
        print(f"Loaded all structures: {len(data_list)} samples")

        n_samples = len(data_list)
        n_val = n_samples // 100
        n_train = n_samples - n_val

        np.random.default_rng(seed=42).shuffle(data_list)
        data_list_train = data_list[:n_train]
        data_list_val = data_list[n_train:]

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data_list_train, self.processed_paths[0])
        torch.save(data_list_val, self.processed_paths[1])


class AFDBStreamingDataset(StructureDataset):
    def __init__(
        self,
        root: "str | Path",
        split: str = "train",
        transform: Optional[Callable] = None,
        n_jobs: int = -1,
    ) -> None:
        self.split = split
        file_idx = {"train": 0, "val": 1}
        self.file_idx = file_idx[split]
        self.n_jobs = n_jobs
        self.root = Path(root)
        self.transform = transform
        self.pdb_dir = self.root / "raw"
        self.processed_dir = self.root / "processed_files"

        if not self.processed_paths[self.file_idx].exists():
            if not self.pdb_dir.exists() or not any(self.pdb_dir.iterdir()):
                from tedbench.utils.io import download_and_extract
                print(f"AFDB raw data not found in {self.root}. Downloading …")
                download_and_extract(_AFDB_URL, self.root, archive_name="afdb.tar.gz")
            self.process()
        self.pdb_files = torch.load(
            self.processed_paths[self.file_idx], weights_only=False
        )

    @property
    def processed_paths(self):
        return [
            self.processed_dir / "train_pdb_files.pt",
            self.processed_dir / "val_pdb_files.pt",
        ]

    def process_func(self, pdb_file: Path) -> Optional[dict]:
        # Process a single PDB file and return a Dictionary
        protein_chain = ProteinChain.from_pdb(pdb_file)
        return self.process_protein_chain(protein_chain)

    @classmethod
    def process_files(cls, pdb_file: Path, max_length: Optional[int] = None) -> Optional[Path]:
        protein_chain = ProteinChain.from_pdb(pdb_file)
        coords, plddt, residue_index = protein_chain.to_structure_encoder_inputs()
        length = len(coords[0])
        max_length = cls.max_length if max_length is None else max_length
        if length > max_length:
            return None
        return pdb_file

    def process(self) -> None:
        # Read data into huge `Data` list.
        pdb_files = sorted(self.pdb_dir.glob("*.pdb"))

        pdb_files = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self.process_files)(pdb_file)
            for pdb_file in track(pdb_files, description="Processing...")
        )

        pdb_files = list(filter(lambda data: data is not None, pdb_files))
        print(f"Loaded all structures: {len(pdb_files)} samples")

        n_samples = len(pdb_files)
        n_val = n_samples // 100
        n_train = n_samples - n_val

        np.random.default_rng(seed=42).shuffle(pdb_files)
        pdb_files_train, pdb_files_val = pdb_files[:n_train], pdb_files[n_train:]

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pdb_files_train, self.processed_paths[0])
        torch.save(pdb_files_val, self.processed_paths[1])

    def __len__(self) -> int:
        return len(self.pdb_files)

    def __getitem__(self, index: int):
        pdb_path = self.pdb_files[index]

        try:
            # Use the processing method that leverages StructureDataset methods
            data = self.process_func(pdb_path)

            if self.transform:
                data = self.transform(data)

            return (
                data["coords"],
                data["residue_index"],
                data["seq_ids"],
                data["protein_chain"],
            )
        except Exception as e:
            print(f"Error processing {pdb_path}: {e}")
            return self[index]

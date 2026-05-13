import joblib
import gzip
import pickle
import numpy as np
import torch
from rich.progress import track
from pathlib import Path
from typing import Literal
from sklearn.model_selection import train_test_split
from minesm.utils.structure.protein_chain import ProteinChain
from .abstract import StructureDataset

_TED_URL = "https://datashare.mpcdf.mpg.de/s/m4owC3SQbd2r6rk/download"


class AFDBTEDStreamingDataset(StructureDataset):
    cath_file_name = "ted_365m.domain_summary.cath.globularity.taxid.tsv.gz"
    chunk_size = 50000000
    cts_cutoff = 10

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        n_jobs=-1,
    ):
        self.split = split
        file_idx = {"train": 0, "val": 1, "test": 2}
        self.file_idx = file_idx[split]
        self.n_jobs = n_jobs
        self.root = Path(root)
        self.transform = transform
        self.pdb_dir = self.root / "raw"
        self.processed_dir = self.root / "processed_files"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.cath_file = self.root / self.cath_file_name
        self.processed_cath_file = self.processed_dir / "afdb_to_cath_ted.pkl"

        if not (self.processed_dir / "cath_labels.pt").exists() and (
            not self.pdb_dir.exists() or not any(self.pdb_dir.iterdir())
        ):
            from tedbench.utils.io import download_and_extract
            print(f"TED raw data not found in {self.root}. Downloading …")
            download_and_extract(_TED_URL, self.root, archive_name="ted.tar.gz")

        if (self.processed_dir / "cath_labels.pt").exists():
            self.load_cath_labels()
        else:
            self.process_cath_file()
            self.load_processed_cath_file()
            self.process_labels()
            self.load_cath_labels()
        self.process()
        cath_data = torch.load(self.processed_paths[self.file_idx], weights_only=False)
        self.pdb_files_split = cath_data["pdb"]
        self.cath_labels = cath_data["labels"]

    def _save_pickle(self, sample_to_cath, chunk_counter):
        # pkl_path = f"{self.processed_cath_file}.{chunk_counter}"
        pkl_path = self.processed_cath_file.with_suffix(f".pkl.{chunk_counter}")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(sample_to_cath, pkl_file)

    def _pickle_exists(self):
        return self.processed_cath_file.with_suffix(".pkl.0").exists()

    def load_processed_cath_file(self):
        """Load pickles from disk and merge them into one dictionary."""
        chunk_counter = 0
        self.sample_to_cath = {}
        if (self.processed_dir / "cath_labels.pt").exists():
            return
        print("Loading CATH mapping data...")
        while self.processed_cath_file.with_suffix(f".pkl.{chunk_counter}").exists():
            print(f"Loading chunk {chunk_counter}...")
            with open(
                self.processed_cath_file.with_suffix(f".pkl.{chunk_counter}"), "rb"
            ) as pkl_file:
                chunk_data = pickle.load(pkl_file)
                self.sample_to_cath.update(chunk_data)
            chunk_counter += 1
        print("CATH mapping data loaded.")

    def process_cath_file(self):
        if self._pickle_exists():
            return
        sample_to_cath = {}
        counter = 0
        chunk_counter = 0
        with gzip.open(self.cath_file, "rt") as file:
            for line in track(file, description="Processing CATH file"):
                parts = line.strip().split("\t")
                full_sample_id = parts[0]
                cath_codes = parts[13].split(",") if parts[13] != "-" else []
                num_residues = parts[4].split(",") if parts[4] != "-" else []
                cath_codes = [
                    (code, int(length))
                    for code, length in zip(cath_codes, num_residues)
                ]
                sample_id = "_".join(full_sample_id.split("_")[:-1])
                if cath_codes:
                    if sample_id not in sample_to_cath:
                        sample_to_cath[sample_id] = []
                        counter += 1
                    sample_to_cath[sample_id].extend(cath_codes)
                if counter == self.chunk_size:
                    self._save_pickle(sample_to_cath, chunk_counter)
                    chunk_counter += 1
                    counter = 0
                    sample_to_cath = {}
        if sample_to_cath:
            self._save_pickle(sample_to_cath, chunk_counter)

    def process_labels(self):
        if (self.processed_dir / "cath_labels.pt").exists():
            return
        pdb_files = sorted(self.pdb_dir.glob("*.pdb"))
        pdb_ids = [pdb_file.with_suffix("").name for pdb_file in pdb_files]
        cath_codes = []
        for pdb_id in pdb_ids:
            _cath_code = self.sample_to_cath.get(pdb_id, [])
            cath_code = []
            for code, num_residues in _cath_code:
                if code.count(".") == 2:
                    code += ".x"
                cath_code.append((code, num_residues))
            cath_codes.append(cath_code)
        indices = []
        filtered_cath_codes = []
        for i, cath_code in enumerate(cath_codes):
            if len(cath_code) > 0:
                indices.append(i)
                cath_code = sorted(cath_code, key=lambda x: x[1], reverse=True)
                filtered_cath_codes.append(cath_code[0][0])
        cath_codes_T = [
            extract_cath_code_by_level(cath_code, "T")
            for cath_code in filtered_cath_codes
        ]
        T_labels, T_inv, T_cts = np.unique(
            cath_codes_T, return_inverse=True, return_counts=True
        )
        A_level_indices = {}
        for i, cath_code in enumerate(T_labels):
            key = ".".join(cath_code.split(".")[:-1])
            if key not in A_level_indices:
                A_level_indices[key] = [i]
            else:
                A_level_indices[key].append(i)
        A_level_indices = {k: np.array(v) for k, v in A_level_indices.items()}
        to_merge_mask = []
        for k, v in A_level_indices.items():
            current_cts = T_cts[v]
            mask = current_cts < self.cts_cutoff
            if mask.sum() == 0:
                to_merge_mask.append(mask)
                continue
            if current_cts[mask].sum() < self.cts_cutoff:
                idx = np.where(mask, float("inf"), current_cts).argmin()
                mask[idx] = True
            to_merge_mask.append(mask)
        to_merge_mask = np.concatenate(to_merge_mask)
        to_merge_mask = to_merge_mask[T_inv]
        cath_codes_updated = [
            ".".join(cath_code.split(".")[:-1]) + ".x"
            if to_merge_mask[i]
            else cath_code
            for i, cath_code in enumerate(cath_codes_T)
        ]
        cath_labels, cath_inv, cath_cts = np.unique(
            cath_codes_updated, return_inverse=True, return_counts=True
        )
        assert np.all(cath_cts >= self.cts_cutoff)
        cath_labels_int = torch.from_numpy(np.arange(len(cath_labels))[cath_inv]).long()
        pdb_files = [pdb_files[i] for i in indices]
        torch.save(cath_labels_int, self.processed_dir / "cath_labels.pt")
        torch.save(pdb_files, self.processed_dir / "filtered_pdb_files.pt")
        torch.save(cath_labels, self.processed_dir / "cath_mapping.pt")

    def load_cath_labels(self):
        self.cath_labels = torch.load(
            self.processed_dir / "cath_labels.pt", weights_only=False
        )
        self.pdb_files = torch.load(
            self.processed_dir / "filtered_pdb_files.pt", weights_only=False
        )
        self.cath_mapping = torch.load(
            self.processed_dir / "cath_mapping.pt", weights_only=False
        )

    @property
    def processed_paths(self):
        return [
            self.processed_dir / "train_ted.pt",
            self.processed_dir / "val_ted.pt",
            self.processed_dir / "test_ted.pt",
        ]

    @property
    def num_classes(self):
        return len(self.cath_mapping)

    def process_func(self, pdb_file):
        # Process a single PDB file and return a Dictionary
        protein_chain = ProteinChain.from_pdb(pdb_file)
        return self.process_protein_chain(protein_chain, max_length=float("inf"))

    def process(self):
        if self.processed_paths[self.file_idx].exists():
            return

        pdb_files = self.pdb_files
        labels = self.cath_labels

        pdb_files_train, pdb_files_test, labels_train, labels_test = train_test_split(
            pdb_files, labels, test_size=0.2, random_state=42, stratify=labels
        )
        pdb_files_val, pdb_files_test, labels_val, labels_test = train_test_split(
            pdb_files_test,
            labels_test,
            test_size=0.5,
            random_state=42,
            stratify=labels_test,
        )

        torch.save(
            {"pdb": pdb_files_train, "labels": labels_train}, self.processed_paths[0]
        )
        torch.save(
            {"pdb": pdb_files_val, "labels": labels_val}, self.processed_paths[1]
        )
        torch.save(
            {"pdb": pdb_files_test, "labels": labels_test}, self.processed_paths[2]
        )

    def __len__(self):
        return len(self.pdb_files_split)

    def __getitem__(self, index: int):
        pdb_path = self.pdb_files_split[index]
        label = self.cath_labels[index]

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
                label,
            )
        except Exception as e:
            print(f"Error processing {pdb_path}: {e}")
            return self[index]


def extract_cath_code_by_level(
    cath_code: str, level: Literal["C", "A", "T", "H"]
) -> str:
    """Extract cath_code at certain level.

    Args:
      cath_code: CATH code.
      level: Level to be extracted

    Returns:
      CATH code at the corresponding level.
    """
    mapping = {"H": 0, "T": 1, "A": 2, "C": 3}
    return cath_code.rsplit(".", mapping[level])[0]

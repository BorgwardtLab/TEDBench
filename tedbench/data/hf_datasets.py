"""HuggingFace-backed dataset classes for TEDBench and CATH 4.4.

These are thin wrappers around the ``datasets`` library that conform to the
same interface as the local loaders (:class:`~tedbench.data.AFDBTEDStreamingDataset`
and :class:`~tedbench.data.CATHTestDataset`).  They slot directly into
:class:`~tedbench.data.LightningStructureDataset` via the ``datasets_map``
and can be selected with a single config change:

.. code-block:: yaml

    # configs/datamodule/hf_ted.yaml
    _target_: tedbench.data.TEDLightningDataset
    root: dexiongc/tedbench   # HF repo ID  — OR —  path to a local HF directory
    dataset_name: hf_ted

The ``root`` parameter is either a HuggingFace repo ID (e.g.
``"dexiongc/tedbench"``) or a local directory previously created with
``save_to_disk`` via ``scripts/upload_datasets.py --save-dir-*``.  Detection
is automatic: if ``Path(root).is_dir()`` the dataset is loaded from disk,
otherwise it is streamed from the Hub.
"""
from pathlib import Path

import torch

from .abstract import StructureDataset


def _load_hf_split(root: str, split: str, cache_dir=None):
    """Load a single split from a local HF directory or the Hub.

    For local paths the dataset must have been saved with ``save_to_disk``:
    - ``DatasetDict`` (TED): splits are subdirectories → loaded via
      ``load_from_disk(root)[split]``
    - Single ``Dataset`` (CATH): the directory IS the dataset → loaded via
      ``load_from_disk(root)``
    """
    from datasets import DatasetDict, load_dataset, load_from_disk

    if Path(root).is_dir():
        ds = load_from_disk(root)
        return ds[split] if isinstance(ds, DatasetDict) else ds
    return load_dataset(root, split=split, cache_dir=cache_dir)


class HFTEDDataset(StructureDataset):
    """TEDBench train / val / test loaded from HuggingFace Hub.

    Args:
        root: HuggingFace repo ID (e.g. ``"dexiongc/tedbench"``).
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``.
        transform: Optional per-sample transform applied in ``__getitem__``.
        cache_dir: Local directory for the HF dataset cache. Defaults to the
            HuggingFace default cache (``~/.cache/huggingface/datasets``).
    """

    def __init__(self, root: str, split: str = "train", transform=None, cache_dir=None):
        self._hf = _load_hf_split(root, split, cache_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int):
        s = self._hf[idx]
        coords        = torch.tensor(s["coords"],        dtype=torch.float32)   # [L, 3, 3]
        residue_index = torch.tensor(s["residue_index"], dtype=torch.int64)
        seq_ids       = torch.tensor(s["seq_ids"],       dtype=torch.int64)
        protein_chain = self.get_protein_chain(
            coords, id=s["name"], sequence=s["sequence"]
        )
        data = dict(
            coords=coords,
            residue_index=residue_index,
            seq_ids=seq_ids,
            protein_chain=protein_chain,
            label=s["label"],
        )
        if self.transform is not None:
            data = self.transform(data)
        return (
            data["coords"],
            data["residue_index"],
            data["seq_ids"],
            data["protein_chain"],
            data["label"],
        )


class HFAFDBDataset(StructureDataset):
    """AFDB pretraining corpus loaded from HuggingFace Hub (no labels).

    Drop-in HF replacement for :class:`~tedbench.data.AFDBStreamingDataset`.
    Returns the same 4-tuple ``(coords, residue_index, seq_ids, protein_chain)``
    so it is compatible with the pretraining Lightning data module.

    The dataset contains 749,679 representative structures from Foldseek-clustered
    AlphaFold Database (pLDDT > 80), with ``"train"`` and ``"val"`` splits.

    Args:
        root: HuggingFace repo ID (e.g. ``"dexiongc/tedbench-afdb"``) or a
            local directory created with ``scripts/upload_datasets.py --save-dir-afdb``.
        split: Dataset split — ``"train"`` or ``"val"``.
        transform: Optional per-sample transform applied in ``__getitem__``.
        cache_dir: Local directory for the HF dataset cache.
    """

    def __init__(self, root: str, split: str = "train", transform=None, cache_dir=None):
        self._hf = _load_hf_split(root, split, cache_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int):
        s = self._hf[idx]
        coords        = torch.tensor(s["coords"],        dtype=torch.float32)
        residue_index = torch.tensor(s["residue_index"], dtype=torch.int64)
        seq_ids       = torch.tensor(s["seq_ids"],       dtype=torch.int64)
        protein_chain = self.get_protein_chain(
            coords, id=s["name"], sequence=s["sequence"]
        )
        data = dict(
            coords=coords,
            residue_index=residue_index,
            seq_ids=seq_ids,
            protein_chain=protein_chain,
        )
        if self.transform is not None:
            data = self.transform(data)
        return (
            data["coords"],
            data["residue_index"],
            data["seq_ids"],
            data["protein_chain"],
        )


class HFCATHTestDataset(StructureDataset):
    """CATH 4.4 experimental test set loaded from HuggingFace Hub.

    Args:
        root: HuggingFace repo ID (e.g. ``"dexiongc/tedbench-cath"``).
        split: Unused; kept for API consistency — the CATH dataset only has a
            ``"test"`` split.
        transform: Optional per-sample transform applied in ``__getitem__``.
        cache_dir: Local directory for the HF dataset cache.
    """

    def __init__(self, root: str, split: str = "test", transform=None, cache_dir=None):
        from datasets import load_dataset
        self._hf = load_dataset(root, split="test", cache_dir=cache_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int):
        s = self._hf[idx]
        coords        = torch.tensor(s["coords"],        dtype=torch.float32)
        residue_index = torch.tensor(s["residue_index"], dtype=torch.int64)
        seq_ids       = torch.tensor(s["seq_ids"],       dtype=torch.int64)
        protein_chain = self.get_protein_chain(
            coords, id=s["name"], sequence=s["sequence"]
        )
        data = dict(
            coords=coords,
            residue_index=residue_index,
            seq_ids=seq_ids,
            protein_chain=protein_chain,
            label=s["label"],
        )
        if self.transform is not None:
            data = self.transform(data)
        return (
            data["coords"],
            data["residue_index"],
            data["seq_ids"],
            data["protein_chain"],
            data["label"],
        )

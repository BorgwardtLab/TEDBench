from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from minesm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from minesm.utils.constants import esm3 as C
from minesm.utils.structure.protein_chain import ProteinChain

from ..utils.tensor_utils import pad_structures


class StructureDataset(Dataset):
    """Base class for protein structure datasets.

    Subclasses populate ``self.data`` with per-sample dicts that include at
    least ``coords``, ``residue_index``, ``seq_ids``, and ``protein_chain``
    keys.  Optional ``label`` key is used for supervised tasks.

    Class attributes:
        max_length (int): Maximum sequence length; longer proteins are dropped
            or truncated during collation (default 512).
        seq_tokenizer: Shared ESM sequence tokenizer for all instances.
    """

    max_length = 512
    seq_tokenizer = EsmSequenceTokenizer()
    data = []
    transform = None

    @classmethod
    def get_protein_chain(cls, bb_coords: torch.Tensor, **kwargs) -> ProteinChain:
        return ProteinChain.from_backbone_atom_coordinates(bb_coords, **kwargs)

    @classmethod
    def process_protein_chain(
        cls, protein_chain: ProteinChain, max_length: Optional[int] = None
    ) -> Optional[dict]:
        coords, plddt, residue_index = protein_chain.to_structure_encoder_inputs()

        max_length = cls.max_length if max_length is None else max_length
        if len(coords[0]) > max_length:
            return None

        is_coord_nan = (
            coords[0][:, :3, :].isnan().any(dim=-1).any(dim=-1)
        )  # [L, 3, 3] -> [L]
        if (~is_coord_nan).count_nonzero() == 0:
            return None
        if is_coord_nan.any():
            protein_chain = protein_chain[~is_coord_nan.numpy()]
            coords, plddt, residue_index = protein_chain.to_structure_encoder_inputs()

        coords = coords[0]
        plddt = plddt[0]
        residue_index = residue_index[0]

        sequence = protein_chain.sequence
        # Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/utils/encoding.py#L48
        if "_" in sequence:
            print("Somehow character - is in protein sequence")
            raise ValueError
        sequence = sequence.replace(C.MASK_STR_SHORT, "<mask>")
        seq_ids = cls.seq_tokenizer.encode(sequence, add_special_tokens=False)
        seq_ids = torch.tensor(seq_ids, dtype=torch.int64)
        assert len(seq_ids) == len(coords)
        return dict(
            coords=coords[:, :3],
            plddt=plddt,
            residue_index=residue_index,
            seq_ids=seq_ids,
            protein_chain=protein_chain,
        )

    def collate_fn(self, batch: list) -> dict:
        """passed to DataLoader as collate_fn argument"""
        batch = list(filter(lambda x: x is not None, batch))

        batch = tuple(zip(*batch))
        label = None
        if len(batch) == 4:
            coords, residue_index, seq_ids, protein_chain = batch
        else:
            coords, residue_index, seq_ids, protein_chain, label = batch

        coords = pad_structures(
            coords,
            constant_value=torch.inf,
            truncation_length=self.max_length,
        )
        residue_index = pad_structures(
            residue_index,
            constant_value=0,
            truncation_length=self.max_length,
            pad_length=coords.shape[1],
        )

        seq_ids = pad_structures(
            seq_ids,
            constant_value=C.SEQUENCE_PAD_TOKEN,
            truncation_length=self.max_length,
            pad_length=coords.shape[1],
        )

        mask = ~(coords[:, :, 0, 0] == torch.inf)
        if label is not None:
            label = torch.tensor(label)

        return dict(
            coords=coords,
            residue_index=residue_index,
            seq_ids=seq_ids,
            protein_chain=protein_chain,
            mask=mask,
            label=label,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)

        if data.get("label", None) is not None:
            return (
                data["coords"],
                data["residue_index"],
                data["seq_ids"],
                data["protein_chain"],
                data["label"],
            )

        return (
            data["coords"],
            data["residue_index"],
            data["seq_ids"],
            data["protein_chain"],
        )

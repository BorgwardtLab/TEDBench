# TEDBench

**TEDBench** is a large-scale, non-redundant benchmark for protein fold classification,
together with **MiAE** (Masked Invariant Autoencoders), a self-supervised pretraining
framework for protein structure representations.

> **Paper:** *Protein Fold Classification at Scale: Benchmarking and Pretraining*  
> Dexiong Chen, Andrei Manolache, Mathias Niepert, Karsten Borgwardt (ICML 2026 spotlight)

---

## Overview

TEDBench is built from the [Encyclopedia of Domains (TED)](https://zenodo.org/records/13908086)
annotations projected onto the Foldseek-clustered AlphaFold Database.

| Split | Structures |
|---|---|
| Train | 369,740 |
| Val | 46,217 |
| Test | 46,218 |
| External test (CATH 4.4 experimental) | 27,638 |

All structures are classified into **965 CATH topology (T-level) classes**.

MiAE is an SE(3)-invariant masked autoencoder that masks up to 90 % of backbone
frames, processes only the visible residues with a geometric encoder, and
reconstructs the full backbone structure with a lightweight decoder.

---

## Installation

**From PyPI (recommended):**

```bash
pip install tedbench
```

**From source** (for training, baselines, or development):

```bash
# 1. Create and activate environment
micromamba create -n tedbench python=3.10 -y
micromamba activate tedbench

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Install the tedbench package (editable)
uv pip install -e .
```

---

## Datasets

Datasets are available from two sources:

| Dataset | HuggingFace | Direct download |
|---|---|---|
| TEDBench (AFDB + CATH labels) | [`TEDBench/ted`](https://huggingface.co/datasets/TEDBench/ted) | [MPCDF datashare](https://datashare.mpcdf.mpg.de/s/m4owC3SQbd2r6rk) |
| AFDB pretraining corpus | [`TEDBench/afdb`](https://huggingface.co/datasets/TEDBench/afdb) | [MPCDF datashare](https://datashare.mpcdf.mpg.de/s/m4owC3SQbd2r6rk) |
| CATH 4.4 experimental test set | [`TEDBench/cath`](https://huggingface.co/datasets/TEDBench/cath) | [MPCDF datashare](https://datashare.mpcdf.mpg.de/s/pjXMpff7GsYTR22) |

The HuggingFace repos require no local setup; the MPCDF archives are auto-downloaded and cached the first time a local dataset class is instantiated (default roots: `./datasets/ted/` and `./datasets/cath/`).

Each sample contains: `coords` `[L, 3, 3]` (backbone N/Cα/C, float32), `plddt` `[L]`, `residue_index` `[L]`, `seq_ids` `[L]`, `sequence`, and `label` (integer CATH topology index).

### Load directly with `datasets`

```python
from datasets import load_dataset
import torch

# TEDBench — train / val / test with CATH labels
ted = load_dataset("TEDBench/ted")
sample = ted["train"][0]
coords    = torch.tensor(sample["coords"])   # [L, 3, 3]
label     = sample["label"]                  # int index
cath_code = ted["train"].features["label"].int2str(label)  # e.g. "3.40.50.300"

# CATH 4.4 external test set
cath = load_dataset("TEDBench/cath", split="test")

# AFDB pretraining corpus
afdb = load_dataset("TEDBench/afdb", split="train")
```

### Use with `LightningStructureDataset`

**From HuggingFace** (`dataset_name="hf_ted"` / `"hf_cath4.4"` / `"hf_afdb"`):

```python
from tedbench.data import LightningStructureDataset

dm = LightningStructureDataset(
    root="TEDBench/ted",   # HF repo ID
    dataset_name="hf_ted",
    batch_size=32,
    num_workers=4,
)
dm.setup("fit")
for batch in dm.train_dataloader():
    print(batch.keys()) 
    # dict_keys(['coords', 'residue_index', 'seq_ids', 'protein_chain', 'mask', 'label'])
```

**Auto-download from MPCDF** (`dataset_name="ted"` / `"cath4.4"` / `"afdb_stream"`): the archive is fetched from the MPCDF datashare and cached under `root` on first use — no manual download needed:

```python
dm = LightningStructureDataset(
    root="./datasets/ted",   # local cache directory
    dataset_name="ted",
    batch_size=32,
    num_workers=4,
)
dm.setup("fit")
for batch in dm.train_dataloader():
    print(batch.keys()) 
    # dict_keys(['coords', 'residue_index', 'seq_ids', 'protein_chain', 'mask', 'label'])
```

Pass `datamodule=hf_ted` (or `datamodule=hf_cath_test`, `datamodule=hf_afdbfs`) to any
training script to use HuggingFace; omit it (or use the default config) for the
auto-downloading local variant.

---

## Pretrained Models

All models are available on HuggingFace and can be loaded with a single call:

```python
from tedbench.utils.io import load_from_hf

model = load_from_hf("TEDBench/miae-b")  # pretrained MiAE-B
model.eval()
```

### Pretrained MiAE (feature extractor / fine-tuning starting point)

| Model | HF repo | Params |
|---|---|---|
| MiAE-S | [`TEDBench/miae-s`](https://huggingface.co/TEDBench/miae-s) | 29 M |
| MiAE-B | [`TEDBench/miae-b`](https://huggingface.co/TEDBench/miae-b) | 102 M |
| MiAE-B+seq | [`TEDBench/miae-b-seq`](https://huggingface.co/TEDBench/miae-b-seq) | 102 M |
| MiAE-L | [`TEDBench/miae-l`](https://huggingface.co/TEDBench/miae-l) | 339 M |

### Fine-tuned on TEDBench (fold classifier)

| Model | HF repo | TEDBench test acc | CATH 4.4 test acc |
|---|---|---|---|
| MiAE-S (ft) | [`TEDBench/miae-s-ft`](https://huggingface.co/TEDBench/miae-s-ft) | 72.28 | 76.08 |
| MiAE-B (ft) | [`TEDBench/miae-b-ft`](https://huggingface.co/TEDBench/miae-b-ft) | 73.71 | 75.72 |
| MiAE-B+seq (ft) | [`TEDBench/miae-b-seq-ft`](https://huggingface.co/TEDBench/miae-b-seq-ft) | 74.56 | 77.34 |
| MiAE-L (ft) | [`TEDBench/miae-l-ft`](https://huggingface.co/TEDBench/miae-l-ft) | 73.47 | 76.46 |

### Trained from scratch on TEDBench (no pretraining)

| Model | HF repo |
|---|---|
| MiAE-S (sc) | [`TEDBench/miae-s-sc`](https://huggingface.co/TEDBench/miae-s-sc) |
| MiAE-B (sc) | [`TEDBench/miae-b-sc`](https://huggingface.co/TEDBench/miae-b-sc) |
| MiAE-B+seq (sc) | [`TEDBench/miae-b-seq-sc`](https://huggingface.co/TEDBench/miae-b-seq-sc) |
| MiAE-L (sc) | [`TEDBench/miae-l-sc`](https://huggingface.co/TEDBench/miae-l-sc) |

---

## Evaluation

Evaluate any model from the HuggingFace Hub without any local data setup:

```bash
# Test fine-tuned MiAE-B on TEDBench test split
python main_test_ted.py \
    datamodule=hf_ted \
    pretrained_model_path=TEDBench/miae-b-ft

# Test on the CATH 4.4 external experimental test set
python main_test_ted.py \
    datamodule=hf_cath_test \
    pretrained_model_path=TEDBench/miae-b-ft

# Test fine-tuned MiAE-B+seq on TEDBench test split
python main_test_ted.py \
    datamodule=hf_ted \
    +model.use_seq_input=true \
    pretrained_model_path=TEDBench/miae-b-seq-ft

# Test supervised-from-scratch MiAE-B
python main_test_ted.py \
    pretrained_model_path=TEDBench/miae-b-sc

# Linear probing with pretrained MiAE-B
python main_linprobe_ted.py \
    pretrained_model_path=TEDBench/miae-b
```

---

## Model Variants

| Name | Params | Layers | Hidden dim | Attn heads |
|---|---|---|---|---|
| `miae_s` | 29 M | 6 | 512 | 8 |
| `miae_b` | 102 M | 12 | 768 | 12 |
| `miae_l` | 339 M | 24 | 1 024 | 16 |

Pass `model.name=<variant>` to any training script to select a size.
Add `model.use_seq_input=true` to enable the **+seq** variant (structure + sequence).

---

## Training and Reproducing Paper Results

See [TRAINING.md](TRAINING.md) for full pretraining, fine-tuning, linear probing,
and baseline reproduction commands with hyperparameter tables.

The `baselines/` directory contains scripts for ESM2, SaProt, and ProteinMPNN baselines.
See [TRAINING.md](TRAINING.md#baselines) for usage.

---

## Citation

```bibtex
@inproceedings{chen2026tedbench,
  title={Protein Fold Classification at Scale: Benchmarking and Pretraining},
  author={Chen, Dexiong and Manolache, Andrei and Niepert, Mathias and Borgwardt, Karsten},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```

---

## License

BSD-3-Clause

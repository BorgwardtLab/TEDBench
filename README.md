# TEDBench

**TEDBench** is a large-scale, non-redundant benchmark for protein fold classification,
together with **MiAE** (Masked Invariant Autoencoders), a self-supervised pretraining
framework for protein structure representations.

> **Paper:** *Protein Fold Classification at Scale: Benchmarking and Pretraining*  
> Dexiong Chen, Andrei Manolache, Mathias Niepert, Karsten Borgwardt (ICML 2026)

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
| TEDBench (AFDB + CATH labels) | [`dexiongc/tedbench`](https://huggingface.co/datasets/dexiongc/tedbench) | [MPCDF datashare](https://datashare.mpcdf.mpg.de/s/m4owC3SQbd2r6rk) |
| AFDB pretraining corpus | [`dexiongc/tedbench-afdb`](https://huggingface.co/datasets/dexiongc/tedbench-afdb) | [MPCDF datashare](https://datashare.mpcdf.mpg.de/s/m4owC3SQbd2r6rk) |
| CATH 4.4 experimental test set | [`dexiongc/tedbench-cath`](https://huggingface.co/datasets/dexiongc/tedbench-cath) | [MPCDF datashare](https://datashare.mpcdf.mpg.de/s/pjXMpff7GsYTR22) |

The HuggingFace repos require no local setup; the MPCDF archives are auto-downloaded and cached the first time a local dataset class is instantiated (default roots: `./datasets/ted/` and `./datasets/cath/`).

Each sample contains: `coords` `[L, 3, 3]` (backbone N/Cα/C, float32), `plddt` `[L]`, `residue_index` `[L]`, `seq_ids` `[L]`, `sequence`, and `label` (integer CATH topology index).

### Load directly with `datasets`

```python
from datasets import load_dataset
import torch

# TEDBench — train / val / test with CATH labels
ted = load_dataset("dexiongc/tedbench")
sample = ted["train"][0]
coords    = torch.tensor(sample["coords"])   # [L, 3, 3]
label     = sample["label"]                  # int index
cath_code = ted["train"].features["label"].int2str(label)  # e.g. "3.40.50.300"

# CATH 4.4 external test set
cath = load_dataset("dexiongc/tedbench-cath", split="test")

# AFDB pretraining corpus
afdb = load_dataset("dexiongc/tedbench-afdb", split="train")
```

### Use with `LightningStructureDataset`

**From HuggingFace** (`dataset_name="hf_ted"` / `"hf_cath4.4"` / `"hf_afdb"`):

```python
from tedbench.data import LightningStructureDataset

dm = LightningStructureDataset(
    root="dexiongc/tedbench",   # HF repo ID
    dataset_name="hf_ted",
    batch_size=32,
    num_workers=4,
)
dm.setup("fit")
for coords, res_idx, seq_ids, chain, label in dm.train_dataloader():
    ...
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
for coords, res_idx, seq_ids, chain, label in dm.train_dataloader():
    ...
```

Pass `datamodule=hf_ted` (or `datamodule=hf_cath_test`, `datamodule=hf_afdbfs`) to any
training script to use HuggingFace; omit it (or use the default config) for the
auto-downloading local variant.

---

## Pretrained Models

All models are available on HuggingFace and can be loaded with a single call:

```python
from tedbench.utils.io import load_from_hf

model = load_from_hf("dexiongc/tedbench-miae-b")  # pretrained MiAE-B
model.eval()
```

### Pretrained MiAE (feature extractor / fine-tuning starting point)

| Model | HF repo | Params |
|---|---|---|
| MiAE-S | [`dexiongc/tedbench-miae-s`](https://huggingface.co/dexiongc/tedbench-miae-s) | 29 M |
| MiAE-B | [`dexiongc/tedbench-miae-b`](https://huggingface.co/dexiongc/tedbench-miae-b) | 102 M |
| MiAE-B+seq | [`dexiongc/tedbench-miae-b-seq`](https://huggingface.co/dexiongc/tedbench-miae-b-seq) | 102 M |
| MiAE-L | [`dexiongc/tedbench-miae-l`](https://huggingface.co/dexiongc/tedbench-miae-l) | 339 M |

### Fine-tuned on TEDBench (fold classifier)

| Model | HF repo | TEDBench test acc | CATH 4.4 test acc |
|---|---|---|---|
| MiAE-S (ft) | [`dexiongc/tedbench-miae-s-ft`](https://huggingface.co/dexiongc/tedbench-miae-s-ft) | 72.28 | 76.08 |
| MiAE-B (ft) | [`dexiongc/tedbench-miae-b-ft`](https://huggingface.co/dexiongc/tedbench-miae-b-ft) | 73.71 | 75.72 |
| MiAE-B+seq (ft) | [`dexiongc/tedbench-miae-b-seq-ft`](https://huggingface.co/dexiongc/tedbench-miae-b-seq-ft) | 74.56 | 77.34 |
| MiAE-L (ft) | [`dexiongc/tedbench-miae-l-ft`](https://huggingface.co/dexiongc/tedbench-miae-l-ft) | 73.47 | 76.46 |

### Trained from scratch on TEDBench (no pretraining)

| Model | HF repo |
|---|---|
| MiAE-S (sc) | [`dexiongc/tedbench-miae-s-sc`](https://huggingface.co/dexiongc/tedbench-miae-s-sc) |
| MiAE-B (sc) | [`dexiongc/tedbench-miae-b-sc`](https://huggingface.co/dexiongc/tedbench-miae-b-sc) |
| MiAE-B+seq (sc) | [`dexiongc/tedbench-miae-b-seq-sc`](https://huggingface.co/dexiongc/tedbench-miae-b-seq-sc) |
| MiAE-L (sc) | [`dexiongc/tedbench-miae-l-sc`](https://huggingface.co/dexiongc/tedbench-miae-l-sc) |

---

## Evaluation

Evaluate any model from the HuggingFace Hub without any local data setup:

```bash
# Test fine-tuned MiAE-B on TEDBench test split
python main_test_ted.py \
    pretrained_model_path=dexiongc/tedbench-miae-b-ft

# Test on the CATH 4.4 external experimental test set
python main_test_ted.py \
    datamodule=hf_cath_test \
    pretrained_model_path=dexiongc/tedbench-miae-b-ft

# Test supervised-from-scratch MiAE-B
python main_test_ted.py \
    pretrained_model_path=dexiongc/tedbench-miae-b-sc

# Linear probing with pretrained MiAE-B
python main_linprobe_ted.py \
    pretrained_model_path=dexiongc/tedbench-miae-b
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

import torch


class Compose:
    """Chain multiple data transforms into a single callable.

    Args:
        transforms: Ordered list of callables, each taking and returning a sample dict.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class RandomNoise:
    """Add Gaussian noise to backbone coordinates as data augmentation.

    Perturbs the ``"coords"`` field of a sample dict.  A standard deviation
    of 0.2 Å is used in the paper (Table 6a/6b).

    Args:
        mean: Mean of the Gaussian noise (Å). Default 0.0.
        std: Standard deviation of the Gaussian noise (Å). Default 0.2.
    """

    def __init__(self, mean=0.0, std=0.2):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        coords = data["coords"]
        noise = torch.randn_like(coords) * self.std + self.mean
        new_data = data.copy()
        new_data["coords"] = coords + noise
        return new_data


class RandomCrop:
    """Randomly crop a protein sequence to a fixed maximum length.

    If the protein has more residues than ``size``, a contiguous window of
    length ``size`` is sampled uniformly.  The crop is applied to the
    ``"coords"`` field; other fields are not cropped (handled during
    collation via padding/truncation).

    Args:
        size: Maximum sequence length after cropping. Default 512.
    """

    def __init__(self, size=512):
        self.size = size

    def __call__(self, data):
        coords = data["coords"]
        if len(coords) > self.size:
            start = torch.randint(0, len(coords) - self.size + 1, (1,))
            data["coords"] = coords[start : start + self.size]
        return data

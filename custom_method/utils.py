import torch
from torch import Tensor


def asymmetry_score(corrupt_to_clean: Tensor, clean_to_corrupt: Tensor):
    assert corrupt_to_clean.shape == clean_to_corrupt.shape, \
        f"Cannot calculate asymmetry between matrices of different shapes, {corrupt_to_clean.shape} and {clean_to_corrupt.shape}"

    # Expect attribution scores in opposite directions to cancel out
    rem_dims = tuple(range(1, len(corrupt_to_clean.shape)))
    max_scores = torch.amax((corrupt_to_clean + clean_to_corrupt), dim=rem_dims, keepdim=True)
    return torch.div((corrupt_to_clean + clean_to_corrupt), max_scores)
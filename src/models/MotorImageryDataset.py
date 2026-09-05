import numpy as np
import torch
from torch.utils.data import Dataset


class MotorImageryDataset(Dataset):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            dtype: torch.dtype = torch.float64,
            domain_labels: np.ndarray | None = None,
    ) -> None:
        if len(x) != len(y):
            raise ValueError("x and y must have matching lengths.")
        if domain_labels is not None and np.shape(domain_labels) != (len(y),):
            raise ValueError("domain_labels must contain one ID per trial.")
        self.x = torch.from_numpy(x).to(dtype=dtype)
        self.y = torch.from_numpy(y).long()
        self.domain_labels = (
            None if domain_labels is None
            else torch.as_tensor(domain_labels, dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        if self.domain_labels is not None:
            return self.x[index], self.y[index], self.domain_labels[index]
        return self.x[index], self.y[index]

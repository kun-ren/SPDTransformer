import numpy as np
import torch
from torch.utils.data import Dataset


class MotorImageryDataset(Dataset):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            dtype: torch.dtype = torch.float64,
    ) -> None:
        self.x = torch.from_numpy(x).to(dtype=dtype)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

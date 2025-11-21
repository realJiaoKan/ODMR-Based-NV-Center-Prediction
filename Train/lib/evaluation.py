import numpy as np
import torch


def delta_vec_norm_evaluator(pred, y):
    diff = pred - y
    norms = torch.linalg.norm(diff, dim=-1)
    return norms.sum().item()


def delta_vec_norm_evaluator_np(pred: np.ndarray, y: np.ndarray) -> float:
    diff = pred - y
    norms = np.linalg.norm(diff, axis=1)
    return float(norms.mean())


if __name__ == "__main__":
    y_true = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 0.0]])
    y_pred = torch.tensor([[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    result = delta_vec_norm_evaluator(y_pred, y_true)
    print(f"Delta vector norm evaluator result: {result}")

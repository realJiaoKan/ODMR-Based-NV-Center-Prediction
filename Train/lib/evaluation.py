import torch


def delta_vec_norm_evaluator(pred, y):
    diff = pred - y
    norms = torch.linalg.norm(diff, dim=-1)
    return norms.sum().item()


if __name__ == "__main__":
    # Test evaluator
    y_true = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 0.0]])
    y_pred = torch.tensor([[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    result = delta_vec_norm_evaluator(y_pred, y_true)
    print(f"Delta vector norm evaluator result: {result}")

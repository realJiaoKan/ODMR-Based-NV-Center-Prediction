import os
from datetime import datetime
from uuid import uuid4

import torch
import torch.optim as optim

from Networks.FFN import Network
from Train.lib.evaluation import delta_vec_norm_evaluator
from Train.lib.train import run

from settings import DEVICE
from Datasets.Spectra import load_loader

BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_RUNS = 3

INPUT_SHAPE = 2000
OUTPUT_SHAPE = 3

HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.1

RESULT_PATH = "Train/Results/FFN"

if __name__ == "__main__":
    train_loader, test_loader = load_loader(BATCH_SIZE)

    for _ in range(NUM_RUNS):
        run_id = str(uuid4())[:8]
        print(f"=== FFN Run {run_id} ===")
        model = Network(
            INPUT_SHAPE,
            OUTPUT_SHAPE,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        log = run(
            model,
            train_loader,
            test_loader,
            optimizer,
            criterion,
            delta_vec_norm_evaluator,
            NUM_EPOCHS,
        )
        # Save results to CSV
        result_file = f"{datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]}_{run_id}.csv"
        with open(os.path.join(RESULT_PATH, result_file), "w") as f:
            f.write("epoch,train_loss,train_eval,test_loss,test_eval\n")
            for epoch_idx, tr_loss, tr_acc, te_loss, te_acc in log:
                f.write(f"{epoch_idx},{tr_loss},{tr_acc},{te_loss},{te_acc}\n")

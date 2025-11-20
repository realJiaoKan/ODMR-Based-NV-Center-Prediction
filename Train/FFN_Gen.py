import torch
import torch.optim as optim

from settings import *
from Train.lib.evaluation import delta_vec_norm_evaluator
from Train.lib.train import run

from Datasets.Spectra import load_loader
from Networks.FFN import Network

BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_RUNS = 3

INPUT_SHAPE = 2000
OUTPUT_SHAPE = 3

RESULT_PATH_PREFIX = "Train/Results/ffn_gen"

if __name__ == "__main__":
    train_loader, test_loader = load_loader(BATCH_SIZE)

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n=== Run {run_idx} ===")
        model = Network(
            INPUT_SHAPE,
            OUTPUT_SHAPE,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
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
        RESULT_PATH = f"{RESULT_PATH_PREFIX}_{run_idx}.csv"
        with open(RESULT_PATH, "w") as f:
            f.write("epoch,train_loss,train_eval,test_loss,test_eval\n")
            for epoch_idx, (tr_loss, tr_acc, te_loss, te_acc) in enumerate(
                log, start=1
            ):
                f.write(f"{epoch_idx},{tr_loss},{tr_acc},{te_loss},{te_acc}\n")

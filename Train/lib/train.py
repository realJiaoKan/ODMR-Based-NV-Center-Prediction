from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

import torch

from settings import *


def train_one_epoch(model, loader, optimizer, criterion, evaluator):
    model.train()
    running_loss = 0
    evaluation = 0
    total = 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Forward pass and backward pass
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # Loss and evaluation
        running_loss += loss.item() * y.size(0)
        evaluation += evaluator(out, y)
        total += y.size(0)

    # Train loss and train evaluation
    return running_loss / total, evaluation / total


def evaluate(model, loader, criterion, evaluator):
    model.eval()
    running_loss = 0
    evaluation = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Forward pass
            out = model(X)
            loss = criterion(out, y)

            # Loss and evaluation
            running_loss += loss.item() * y.size(0)
            evaluation += evaluator(out, y)
            total += y.size(0)

    # Test loss and test evaluation
    return running_loss / total, evaluation / total


def run(model, train_loader, test_loader, optimizer, criterion, evaluator, epochs):
    log = []
    for epoch in range(1, epochs + 1):
        # Training
        print("Training...", end=" ", flush=True)
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, evaluator
        )

        # Evaluation
        print("Evaluating...", end=" ", flush=True)
        te_loss, te_acc = evaluate(model, test_loader, criterion, evaluator)

        # Logging
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f}, train_eval={tr_acc:.4f}, "
            f"test_loss={te_loss:.4f}, test_eval={te_acc:.4f}"
        )
        log.append((epoch, tr_loss, tr_acc, te_loss, te_acc))

    return log


def run_sklearn_ml(model, X_train, y_train, X_test, y_test, evaluator):
    with tqdm_joblib(desc="Training Trees", total=model.n_estimators):
        model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_eval = evaluator(train_pred, y_train)
    test_eval = evaluator(test_pred, y_test)

    print(f"train_eval: {train_eval:.4f}, test_eval: {test_eval:.4f}")
    return train_eval, test_eval

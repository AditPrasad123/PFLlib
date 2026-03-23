import torch
import numpy as np
import gc

from feature_extraction import extract_features
from qiskit_qsvm import get_qsvm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SAMPLES = 700
MAX_TEST = 500

def run_qsvm_on_clients(path, tag):
    print(f"\n===== QSVM on {tag} =====")

    clients = torch.load(path, weights_only=False)
    accs = []

    for client in clients:
        print(f"\nClient {client.id}")

        train_loader = client.load_train_data()
        test_loader = client.load_test_data()

        X_train, y_train = extract_features(client.model, train_loader, device)
        X_test, y_test = extract_features(client.model, test_loader, device)

        X_train, X_test = X_train.float(), X_test.float()
        
        if len(X_train) > MAX_SAMPLES:
            idx = np.random.choice(len(X_train), MAX_SAMPLES, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]

        if len(X_test) > MAX_TEST:
            idx = np.random.choice(len(X_test), MAX_TEST, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]

        clf = get_qsvm(num_features=X_train.shape[1])

        print("Training QSVM...")
        clf.fit(X_train.numpy(), y_train.numpy())
        acc = clf.score(X_test.numpy(), y_test.numpy())
        print("Finished QSVM training...")
        print(f"QSVM Accuracy: {acc:.4f}")
        accs.append(acc)
        del X_train, y_train, X_test, y_test, clf
        gc.collect()

        
    print(f"\n--- {tag} Summary ---")
    print(f"Mean Accuracy: {np.mean(accs):.4f}")
    print(f"Std Dev: {np.std(accs):.4f}")


# 🔥 RUN BOTH
run_qsvm_on_clients("clients_finetune.pt", "FedBABU (Fine-tuned)")
# run_qsvm_on_clients("clients_ttft.pt", "FedBABU + TTFT")
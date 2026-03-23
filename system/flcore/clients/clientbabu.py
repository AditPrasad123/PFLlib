from http import client

import numpy as np
import time
import torch
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from flcore.utils.metrics import MetricsCalculator


def _softmax_np(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


class clientBABU(Client):
    """
    FedBABU (Body Aggregation, Body Update) Client Implementation
    
    FedBABU is a personalized federated learning algorithm where:
    - The BODY (backbone CNN + transformer features): Trained with updates sent to server for global aggregation
    - The HEAD (classifier): Locally personalized and NOT sent to server
    
    This strategy allows clients to adapt to local data distributions while maintaining
    shared feature learning across the federation.
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Duration (in local epochs) for fine-tuning head on local data after global training
        self.fine_tuning_epochs = 10

        # Initialize optimizer: Only head parameters are trainable during initialization
        # The backbone will be unfrozen during local training to enable body updates
        self.optimizer = torch.optim.SGD(
                self.model.head.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4
            )

    def train(self):
        for p in self.model.base.parameters():
            p.requires_grad = True
        for p in self.model.head.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )

        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)
                loss = self.loss(out, y)   # uses CrossEntropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    # Explicitly expose metrics to avoid missing attribute errors
    def test_metrics(self):
        return super().test_metrics()

    def train_metrics(self):
        return super().train_metrics()

    def set_parameters(self, model):
        """
        Receive global BODY updates from server.
        
        FedBABU Protocol: Server aggregates and sends back the updated BODY (CNN + transformer)
        from all clients. This method applies those global updates locally, preserving the
        locally-trained HEAD which is not shared in the federation.
        """
        # model may already be the backbone (FedBABU behavior)
        src = model.base if hasattr(model, "base") else model
        tgt = self.model.base

        for new_param, old_param in zip(src.parameters(), tgt.parameters()):
            old_param.data = new_param.data.clone()



    def fine_tune(self):
        trainloader = self.load_train_data()
        self.model.train()

        for p in self.model.base.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            self.model.head.parameters(),
            lr=self.learning_rate * 5,
            weight_decay=1e-4
        )

        for _ in range(self.fine_tuning_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.loss(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test_time_finetune(self, epochs=5, lr=1e-3):
        self.model.train()

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.model.head.parameters(), lr=lr)
        loader = self.load_test_data(batch_size=16)

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.loss(out, y)   # uses CrossEntropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

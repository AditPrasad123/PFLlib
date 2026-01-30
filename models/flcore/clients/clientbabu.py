import numpy as np
import time
import torch
from flcore.clients.clientbase import Client


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
        """
        Local training step for FedBABU client.
        
        Key FedBABU Characteristic: Updates to the BODY (CNN + transformer) are computed
        and sent to the server for global aggregation. This ensures shared feature learning.
        """
        # ============ Part 1: Backbone Parameter Configuration ============
        # Freeze early CNN layers (first 4 blocks) to preserve learned low-level features
        # These early features (edges, textures) generalize well across clients
        for p in self.model.base.cnn[:4].parameters():
            p.requires_grad = False

        # Enable training on later CNN blocks + transformer + head
        # These layers capture high-level features specific to this client's local data
        for p in self.model.base.cnn[4:].parameters():
            p.requires_grad = True
        for p in self.model.base.transformer.parameters():
            p.requires_grad = True
        for p in self.model.head.parameters():
            p.requires_grad = True
        
        # Create optimizer over all trainable parameters (body parts + head)
        # This ensures the BODY receives gradient updates that will be aggregated globally
        self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=1e-4
                )
        trainloader = self.load_train_data()
        
        start_time = time.time()
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # ============ Part 2: Standard Training Loop ============
        # Train body (CNN + transformer) and head jointly on local data
        # Body updates will be aggregated globally; head updates stay local
        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.loss(out, y)

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

    def set_parameters(self, base):
        """
        Receive global BODY updates from server.
        
        FedBABU Protocol: Server aggregates and sends back the updated BODY (CNN + transformer)
        from all clients. This method applies those global updates locally, preserving the
        locally-trained HEAD which is not shared in the federation.
        """
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def fine_tune(self):
        """
        Post-training personalization phase.
        
        After global training rounds, fine-tune ONLY the HEAD using aggressive learning.
        The BODY is completely frozen to prevent overfitting to local data.
        This personalizes the classifier to client-specific data distributions.
        """
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        # Completely freeze entire body (no shared learning during personalization)
        for p in self.model.base.parameters():
            p.requires_grad = False

        # Only head is trainable for personalization
        for p in self.model.head.parameters():
            p.requires_grad = True

        # Use AdamW with aggressive learning rate (5x base rate) for rapid adaptation
        # This helps the head quickly adapt to client-specific label distributions
        optimizer = torch.optim.AdamW(
            self.model.head.parameters(),
            lr=self.learning_rate * 5,   # Aggressive learning rate for personalization
            weight_decay=1e-4
        )

        for epoch in range(self.fine_tuning_epochs):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)

                out = self.model(x)
                loss = self.loss(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_time_finetune(self, epochs=5, lr=1e-3):
        """
        Optional fine-tuning during test/evaluation phase.
        
        Allows head-only adaptation on test data before evaluation.
        Useful for measuring generalization when clients can adapt at test time.
        """
        self.model.train()

        # Freeze entire body to maintain shared feature representation
        for p in self.model.parameters():
            p.requires_grad = False
        # Only head can adapt to test-time data distribution
        for p in self.model.head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.model.head.parameters(), lr=lr)
        loss_fn = self.loss

        loader = self.load_test_data(batch_size=16)

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = loss_fn(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
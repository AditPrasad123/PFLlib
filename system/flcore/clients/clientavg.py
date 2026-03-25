import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fine_tuning_epochs = args.fine_tuning_epochs

    def _get_head_module(self):
        for name in ("head", "fc", "classifier"):
            if hasattr(self.model, name):
                return getattr(self.model, name)
        return None

    def _move_batch_to_device(self, x, y):
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
            x = x[0]
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if epoch == 0 and i == 0:
                    print(f"[Client {self.id}] batch x:", x.shape, "y:", y.shape)
                    print(f"[Client {self.id}] output:", output.shape)
        
        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def fine_tune(self):
        head = self._get_head_module()
        if head is None:
            return

        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=self.learning_rate * 5,
            weight_decay=1e-4,
        )

        trainloader = self.load_train_data()
        for _ in range(self.fine_tuning_epochs):
            for x, y in trainloader:
                x, y = self._move_batch_to_device(x, y)
                out = self.model(x)
                loss = self.loss(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test_time_finetune(self, epochs=5, lr=1e-3):
        head = self._get_head_module()
        if head is None:
            return

        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        loader = self.load_test_data(batch_size=16)

        for _ in range(epochs):
            for x, y in loader:
                x, y = self._move_batch_to_device(x, y)
                out = self.model(x)
                loss = self.loss(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

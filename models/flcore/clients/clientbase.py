import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score
)

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # Imbalance handling options
        self.use_class_weight = getattr(args, 'class_weighted_loss', False)
        self.use_focal_loss = getattr(args, 'focal_loss', False)
        self.focal_gamma = getattr(args, 'focal_gamma', 2.0)
        self.label_smoothing = getattr(args, 'label_smoothing', 0.0)
        self.use_oversample = getattr(args, 'oversample', False)

        # Compute per-class weights from this client's labels if requested
        class_weights_tensor = None
        if self.use_class_weight or self.use_focal_loss:
            try:
                train_data_for_stats = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
                labels_array = None
                if hasattr(train_data_for_stats, 'labels'):
                    labels_array = np.array(train_data_for_stats.labels)
                elif isinstance(train_data_for_stats, Subset) and hasattr(train_data_for_stats.dataset, 'labels'):
                    base_labels = np.array(train_data_for_stats.dataset.labels)
                    labels_array = base_labels[np.array(train_data_for_stats.indices)]

                if labels_array is not None and labels_array.size > 0:
                    counts = np.bincount(labels_array, minlength=self.num_classes)
                    counts = np.where(counts == 0, 1, counts)
                    weights = np.sqrt((labels_array.size) / (self.num_classes * counts))
                    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            except Exception:
                class_weights_tensor = None

        if self.use_focal_loss:
            self.loss = FocalLoss(alpha=class_weights_tensor, gamma=self.focal_gamma)
        else:
            if self.label_smoothing and self.label_smoothing > 0.0:
                try:
                    self.loss = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=self.label_smoothing)
                except TypeError:
                    self.loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
            else:
                self.loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
        # # Only include parameters that require gradients (useful when backbone is frozen)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            momentum=args.momentum
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)

        if self.use_oversample:
            labels_array = None
            if hasattr(train_data, 'labels'):
                labels_array = np.array(train_data.labels)
            elif isinstance(train_data, Subset) and hasattr(train_data.dataset, 'labels'):
                base_labels = np.array(train_data.dataset.labels)
                labels_array = base_labels[np.array(train_data.indices)]

            if labels_array is not None and labels_array.size > 0:
                class_count = np.bincount(labels_array, minlength=self.num_classes)
                class_count = np.where(class_count == 0, 1, class_count)
                class_weights = (labels_array.size) / (self.num_classes * class_count)
                sample_weights = class_weights[labels_array]
                sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double),
                                                len(sample_weights), replacement=True)
                return DataLoader(train_data, batch_size, drop_last=True, sampler=sampler)

        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()


    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        #print("Shape of y_true:", y_true.shape, "Shape of y_prob:", y_prob.shape)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc
    
    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1, 1)).squeeze(1)
        focal_factor = (1 - pt).pow(self.gamma)
        loss = focal_factor * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
    def test_time_finetune(self):
        self.model.train()

        # Freeze everything except classifier
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.SGD(
            self.model.head.parameters(),
            lr=1e-3,
            momentum=0.9
        )

        loader = self.load_train_data(batch_size=8)

        for _ in range(5):  # TTFT epochs
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss(self.model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

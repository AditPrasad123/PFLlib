import numpy as np
import time
import torch
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

        self.use_kernel_classifier = getattr(args, "use_kernel_classifier", False)
        self.kernel_classifier_type = getattr(args, "kernel_classifier_type", "quantum_kernel_svm")
        self.kernel_max_train_samples = getattr(args, "kernel_max_train_samples", 600)
        self.kernel_gamma = getattr(args, "kernel_gamma", 0.5)
        self.kernel_q_layers = getattr(args, "kernel_q_layers", 2)
        self.kernel_q_shots = getattr(args, "kernel_q_shots", 0)
        self.kernel_classifier = None
        self.kernel_train_support = None
        self.kernel_scaler_mean = None
        self.kernel_scaler_std = None

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

    def _extract_embeddings(self, x):
        if hasattr(self.model, "extract_embeddings"):
            return self.model.extract_embeddings(x)
        return self.model.base(x)

    def _collect_embeddings_labels(self, train=True, max_samples=None):
        loader = self.load_train_data() if train else self.load_test_data()
        self.model.eval()

        all_emb = []
        all_y = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                emb = self._extract_embeddings(x)
                all_emb.append(emb.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())

        if len(all_emb) == 0:
            return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

        X = np.concatenate(all_emb, axis=0).astype(np.float32)
        y = np.concatenate(all_y, axis=0).astype(np.int64)

        if max_samples is not None and len(y) > max_samples:
            # Class-balanced-ish capped sampling for tractable kernel complexity.
            selected = []
            classes = np.unique(y)
            per_class = max(1, max_samples // max(1, len(classes)))
            rng = np.random.default_rng(seed=42 + self.id)
            for c in classes:
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                selected.extend(idx[:per_class].tolist())
            if len(selected) < max_samples:
                remaining = np.setdiff1d(np.arange(len(y)), np.array(selected, dtype=np.int64), assume_unique=False)
                rng.shuffle(remaining)
                selected.extend(remaining[: max_samples - len(selected)].tolist())
            selected = np.array(selected[:max_samples], dtype=np.int64)
            X = X[selected]
            y = y[selected]

        return X, y

    def _standardize_embeddings(self, X, fit=False):
        if fit or self.kernel_scaler_mean is None or self.kernel_scaler_std is None:
            self.kernel_scaler_mean = X.mean(axis=0, keepdims=True)
            self.kernel_scaler_std = X.std(axis=0, keepdims=True)
            self.kernel_scaler_std[self.kernel_scaler_std < 1e-6] = 1.0
        return (X - self.kernel_scaler_mean) / self.kernel_scaler_std

    def _fit_classical_svm(self, X_train, y_train):
        from sklearn.svm import SVC

        clf = SVC(kernel='rbf', gamma=self.kernel_gamma, probability=True, class_weight='balanced')
        clf.fit(X_train, y_train)
        self.kernel_classifier = clf
        self.kernel_train_support = None

    def _fit_quantum_kernel_svm(self, X_train, y_train):
        from sklearn.svm import SVC

        try:
            import pennylane as qml
        except Exception:
            # Safe fallback if quantum dependency is unavailable.
            self._fit_classical_svm(X_train, y_train)
            self.kernel_classifier_type = "svm_rbf"
            return

        n_qubits = X_train.shape[1]
        if n_qubits > 8:
            # Keep quantum kernel tractable in practical runs.
            X_train = X_train[:, :8]
            n_qubits = 8

        if self.kernel_q_shots and self.kernel_q_shots > 0:
            dev = qml.device("default.qubit", wires=n_qubits, shots=int(self.kernel_q_shots))
        else:
            dev = qml.device("default.qubit", wires=n_qubits)

        wires = list(range(n_qubits))
        q_layers = max(1, int(self.kernel_q_layers))

        def feature_map(x):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            for _ in range(q_layers):
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])

        @qml.qnode(dev)
        def kernel_qnode(x1, x2):
            feature_map(x1)
            qml.adjoint(feature_map)(x2)
            return qml.probs(wires=wires)

        def quantum_kernel(x1, x2):
            return float(kernel_qnode(x1, x2)[0])

        n = X_train.shape[0]
        K_train = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            K_train[i, i] = 1.0
            for j in range(i + 1, n):
                kval = quantum_kernel(X_train[i], X_train[j])
                K_train[i, j] = kval
                K_train[j, i] = kval

        clf = SVC(kernel='precomputed', probability=True, class_weight='balanced')
        clf.fit(K_train, y_train)

        self.kernel_classifier = clf
        self.kernel_train_support = X_train
        self._quantum_kernel_callable = quantum_kernel

    def _predict_kernel_classifier(self, X_test):
        if self.kernel_classifier is None:
            raise RuntimeError("Kernel classifier is not initialized.")

        if self.kernel_classifier_type == "quantum_kernel_svm" and self.kernel_train_support is not None:
            X_ref = self.kernel_train_support
            X_eval = X_test
            if X_ref.shape[1] != X_eval.shape[1]:
                d = min(X_ref.shape[1], X_eval.shape[1])
                X_ref = X_ref[:, :d]
                X_eval = X_eval[:, :d]

            K_test = np.zeros((X_eval.shape[0], X_ref.shape[0]), dtype=np.float64)
            for i in range(X_eval.shape[0]):
                for j in range(X_ref.shape[0]):
                    K_test[i, j] = self._quantum_kernel_callable(X_eval[i], X_ref[j])

            if hasattr(self.kernel_classifier, "predict_proba"):
                y_prob = self.kernel_classifier.predict_proba(K_test)
            else:
                logits = self.kernel_classifier.decision_function(K_test)
                if logits.ndim == 1:
                    logits = np.stack([-logits, logits], axis=1)
                y_prob = _softmax_np(logits)
            y_pred = np.argmax(y_prob, axis=1)
            return y_pred.astype(np.int64), y_prob.astype(np.float64)

        if hasattr(self.kernel_classifier, "predict_proba"):
            y_prob = self.kernel_classifier.predict_proba(X_test)
        else:
            logits = self.kernel_classifier.decision_function(X_test)
            if logits.ndim == 1:
                logits = np.stack([-logits, logits], axis=1)
            y_prob = _softmax_np(logits)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred.astype(np.int64), y_prob.astype(np.float64)

    def train(self):
        """
        Local training step for FedBABU client.
        
        Key FedBABU Characteristic: Updates to the BODY (CNN + transformer) are computed
        and sent to the server for global aggregation. This ensures shared feature learning.
        """
        for p in self.model.base.parameters():
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
        """
        Post-training personalization phase.
        
        After global training rounds, fine-tune ONLY the HEAD using aggressive learning.
        The BODY is completely frozen to prevent overfitting to local data.
        This personalizes the classifier to client-specific data distributions.
        """
        if self.use_kernel_classifier:
            # Build a local kernel classifier over frozen embeddings for personalization.
            self.model.eval()
            for p in self.model.base.parameters():
                p.requires_grad = False

            cap = int(self.kernel_max_train_samples)
            if self.kernel_classifier_type == "quantum_kernel_svm":
                cap = min(cap, 200)

            X_train, y_train = self._collect_embeddings_labels(train=True, max_samples=cap)
            if X_train.shape[0] < 2 or len(np.unique(y_train)) < 2:
                return

            X_train = self._standardize_embeddings(X_train, fit=True)

            if self.kernel_classifier_type == "svm_rbf":
                self._fit_classical_svm(X_train, y_train)
            else:
                self._fit_quantum_kernel_svm(X_train, y_train)
            return

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
        if self.use_kernel_classifier:
            # For kernel classifiers, keep TTFT disabled to avoid test-set leakage and high kernel cost.
            return

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

    def test_metrics(self):
        if not self.use_kernel_classifier or self.kernel_classifier is None:
            return super().test_metrics()

        X_test, y_true = self._collect_embeddings_labels(train=False, max_samples=None)
        if X_test.shape[0] == 0:
            return 0.0, 0, 0.0

        X_test = self._standardize_embeddings(X_test, fit=False)
        y_pred, y_prob = self._predict_kernel_classifier(X_test)

        if y_prob.shape[1] != self.num_classes:
            aligned = np.zeros((y_prob.shape[0], self.num_classes), dtype=np.float64)
            classes_ = getattr(self.kernel_classifier, "classes_", np.arange(y_prob.shape[1]))
            for idx, cls in enumerate(classes_):
                if 0 <= int(cls) < self.num_classes:
                    aligned[:, int(cls)] = y_prob[:, idx]
            row_sum = aligned.sum(axis=1, keepdims=True)
            row_sum[row_sum <= 0] = 1.0
            y_prob = aligned / row_sum

        test_acc = int(np.sum(y_pred == y_true))
        test_num = int(y_true.shape[0])
        auc = self._compute_auc_roc(y_true, y_prob)
        return test_acc, test_num, auc

    def test_metrics_detailed(self):
        if not self.use_kernel_classifier or self.kernel_classifier is None:
            return super().test_metrics_detailed()

        X_test, y_true = self._collect_embeddings_labels(train=False, max_samples=None)
        if X_test.shape[0] == 0:
            return {}

        X_test = self._standardize_embeddings(X_test, fit=False)
        y_pred, y_prob = self._predict_kernel_classifier(X_test)

        # Ensure probability matrix has all classes in fixed order [0..C-1]
        if y_prob.shape[1] != self.num_classes:
            aligned = np.zeros((y_prob.shape[0], self.num_classes), dtype=np.float64)
            classes_ = getattr(self.kernel_classifier, "classes_", np.arange(y_prob.shape[1]))
            for idx, cls in enumerate(classes_):
                if 0 <= int(cls) < self.num_classes:
                    aligned[:, int(cls)] = y_prob[:, idx]
            row_sum = aligned.sum(axis=1, keepdims=True)
            row_sum[row_sum <= 0] = 1.0
            y_prob = aligned / row_sum

        calc = MetricsCalculator(num_classes=self.num_classes)
        detailed = calc.calculate_classification_metrics(y_true, y_pred, y_prob)
        detailed['test_samples'] = int(y_true.shape[0])
        detailed['test_correct'] = int(np.sum(y_pred == y_true))
        return detailed
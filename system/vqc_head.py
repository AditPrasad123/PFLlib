"""VQC head components for quantum-enhanced classifiers."""
import torch
import torch.nn as nn
import pennylane as qml


class VQCBlock(nn.Module):
    """A small variational quantum circuit that maps R^n -> R^n via Z readout."""
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.layer(x)


class VQCHead(nn.Module):
    """Project features to qubit angles, run VQC, and classify."""
    def __init__(self, in_features, num_classes, n_qubits=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(in_features, n_qubits)
        self.vqc = VQCBlock(n_qubits=n_qubits, n_layers=n_layers)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.vqc(x)
        return self.classifier(x)

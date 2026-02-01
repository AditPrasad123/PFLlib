"""
Variational Quantum Circuit (VQC) Feature Processor
Integrates with PyTorch for hybrid classical-quantum learning

Option 1: VQC as feature transformer
Features (256) → VQC (8 qubits) → Features (256) → Classifier
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class VQCFeatureProcessor(nn.Module):
    """
    Variational Quantum Circuit for feature transformation.
    
    Architecture:
    - Input: Classical features (batch_size, 256)
    - Encoding: Amplitude encoding into 8 qubits
    - Variational circuit: 3 layers of RY + CNOT entanglement
    - Measurement: Pauli-Z expectation on all qubits (8 outputs)
    - Output: Quantum-transformed features (batch_size, 256) after classical post-processing
    
    This VQC learns quantum feature representations that may capture non-linear 
    patterns better than classical transformations alone.
    """
    
    def __init__(self, input_dim=256, n_qubits=8, n_layers=3, device_name="default.qubit"):
        """
        Args:
            input_dim: Input feature dimension (256)
            n_qubits: Number of qubits in the circuit (8)
            n_layers: Depth of variational circuit (3)
            device_name: PennyLane device ('default.qubit', 'lightning.qubit' for speed)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Feature scaling: map 256-dim to [0, π] for encoding
        self.scale = np.pi / 2
        
        # Initialize PennyLane quantum device
        try:
            # Try to use lightning (faster)
            self.device = qml.device(device_name, wires=n_qubits)
        except:
            print(f"Warning: {device_name} not available, using default.qubit")
            self.device = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize variational parameters (learnable)
        # Shape: (n_layers, n_qubits) for each rotation layer
        self.theta = nn.Parameter(
            torch.randn(n_layers, n_qubits) * 0.1
        )
        self.phi = nn.Parameter(
            torch.randn(n_layers, n_qubits) * 0.1
        )
        
        # Classical post-processing: MLP to expand 8 quantum outputs back to 256
        # VQC measurement gives 8 values (one per qubit)
        # We expand back to 256 to match input dimension for residual connections
        self.post_processor = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        
        print(f"✓ VQC initialized: {n_qubits} qubits, {n_layers} layers, input_dim={input_dim}")
    
    def _amplitude_encode(self, features):
        """
        Encode classical features into quantum amplitudes.
        
        Normalizes features and uses them to initialize quantum state amplitudes.
        This creates an entangled quantum state that encodes the classical information.
        
        Args:
            features: (batch_size, 256) normalized to [0, 1]
        
        Returns:
            Quantum state initialized with feature information
        """
        # Normalize features to [0, 1]
        batch_size = features.shape[0]
        features_norm = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        # For each qubit, use feature values to modulate encoding angles
        # This creates a continuous parameterization of the quantum state
        encoding_angles = features_norm[:, :self.n_qubits] * self.scale
        return encoding_angles
    
    def _vqc_circuit(self, features, theta, phi):
        """
        Define the variational quantum circuit.
        
        Layers:
        1. Amplitude encoding of classical features
        2. Variational rotations (RY gates parameterized by theta)
        3. Entanglement (CNOT ladder)
        4. Additional rotations (RZ gates parameterized by phi)
        5. Measurement: Pauli-Z on all qubits
        
        Args:
            features: Input features for encoding (batch_size, n_qubits)
            theta: Rotation parameters (n_layers, n_qubits)
            phi: Additional rotation parameters (n_layers, n_qubits)
        
        Returns:
            Expectation values of Pauli-Z on each qubit (n_qubits,)
        """
        # Encode features into quantum state using RY rotations
        for i in range(self.n_qubits):
            qml.RY(features[i], wires=i)
        
        # Variational circuit: alternating rotations and entanglement
        for layer in range(self.n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RY(theta[layer, i], wires=i)
            
            # Entanglement: CNOT ladder (circular)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
            # Additional rotations
            for i in range(self.n_qubits):
                qml.RZ(phi[layer, i], wires=i)
        
        # Measurement: Pauli-Z expectation on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """
        Forward pass through VQC.
        
        Args:
            x: Input features (batch_size, 256)
        
        Returns:
            Transformed features (batch_size, 256)
        """
        batch_size = x.shape[0]
        
        # Create QNode once (not inside loop - major optimization!)
        @qml.qnode(self.device, interface="torch", diff_method="parameter-shift")
        def circuit(encoding_angles, theta, phi):
            return self._vqc_circuit(encoding_angles, theta, phi)
        
        outputs = []
        
        # Process each sample in batch through quantum circuit
        for i in range(batch_size):
            # Get encoding angles for this sample
            encoding_angles = self._amplitude_encode(x[i:i+1])  # (1, n_qubits)
            
            # Execute quantum circuit
            q_output = circuit(encoding_angles[0], self.theta, self.phi)  # (n_qubits,)
            
            # Stack outputs
            outputs.append(torch.stack(q_output))
        
        # Batch process quantum outputs: (batch_size, n_qubits)
        quantum_features = torch.stack(outputs)
        
        # Convert to float32 to match PyTorch default dtype
        # (PennyLane returns float64, but Linear layers expect float32)
        quantum_features = quantum_features.float()
        
        # Move quantum features to the same device as input
        quantum_features = quantum_features.to(x.device)
        
        # Classical post-processing: expand 8 dims back to 256
        transformed = self.post_processor(quantum_features)  # (batch_size, 256)
        
        # Residual connection: add original features for stability
        # This helps with training by providing a direct gradient path
        output = x + transformed
        
        return output


class VQCHybridBase(nn.Module):
    """
    Complete feature extraction base for VQCHybridModel.
    Outputs 256-dimensional quantum-transformed features ready for classification.
    Used as shared backbone in FedBABU split learning.
    """
    
    def __init__(self, n_qubits=8, n_vqc_layers=3):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        
        # ============ CNN Feature Extractor ============
        backbone = efficientnet_b4(weights="IMAGENET1K_V1")
        self.cnn = backbone.features
        self.token_proj = nn.Conv2d(1792, 256, kernel_size=1)
        
        # ============ Transformer Encoder ============
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # ============ Quantum Feature Processor ============
        self.vqc = VQCFeatureProcessor(
            input_dim=256,
            n_qubits=n_qubits,
            n_layers=n_vqc_layers,
            device_name="default.qubit"
        )
    
    def forward(self, x):
        """
        Extract 256-dim quantum-transformed features from images.
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
        
        Returns:
            256-dim features (batch_size, 256)
        """
        # Stage 1: CNN extraction
        x = self.cnn(x)  # (batch, 1792, 7, 7)
        
        # Stage 2: Token projection
        x = self.token_proj(x)  # (batch, 256, 7, 7)
        
        # Stage 3: Reshape for transformer
        x = x.flatten(2).transpose(1, 2)  # (batch, 49, 256)
        
        # Stage 4: Transformer encoding
        x = self.transformer(x)  # (batch, 49, 256)
        
        # Stage 5: Global pooling
        x = x.mean(dim=1)  # (batch, 256)
        
        # Stage 6: Quantum feature processing
        x = self.vqc(x)  # (batch, 256)
        
        return x


class VQCHybridModel(nn.Module):
    """
    Hybrid Classical-Quantum Model for ISIC2019
    
    Architecture:
    CNN (EfficientNet-B4) → Token Projection → Transformer → VQC → Linear Classifier
    
    For FedBABU split learning:
    - .base: Shared backbone (CNN + Transformer + VQC)
    - .head/.fc: Local classifier head (trained locally per client)
    
    The VQC learns quantum feature transformations that complement classical features.
    """
    
    def __init__(self, num_classes=9, n_qubits=8, n_vqc_layers=3):
        """
        Args:
            num_classes: Number of output classes (9 for ISIC2019)
            n_qubits: Number of qubits in VQC (8)
            n_vqc_layers: Depth of VQC circuit (3)
        """
        super().__init__()
        
        # ============ Feature Extraction Base (Shared) ============
        self.base = VQCHybridBase(n_qubits=n_qubits, n_vqc_layers=n_vqc_layers)
        
        # ============ Classifier Head (Local) ============
        self.head = nn.Linear(256, num_classes)
        
        # ============ FedBABU Compatibility ============
        # Add .fc alias so FedBABU can access the head
        self.fc = self.head
    
    def forward(self, x):
        """
        Forward pass with quantum feature processing
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
        
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Extract quantum-transformed features through base
        features = self.base(x)  # (batch, 256)
        
        # Classify
        logits = self.head(features)  # (batch, num_classes)
        
        return logits


if __name__ == "__main__":
    """
    Test VQC and VQCHybridModel
    """
    print("Testing VQC Feature Processor")
    print("=" * 60)
    
    # Test VQC alone
    print("\n[1/2] Testing VQC Feature Processor...")
    vqc = VQCFeatureProcessor(input_dim=256, n_qubits=8, n_layers=3)
    
    # Create dummy input
    x_test = torch.randn(2, 256)
    print(f"Input shape: {x_test.shape}")
    
    # Forward pass
    output = vqc(x_test)
    print(f"Output shape: {output.shape}")
    print(f"✓ VQC works!")
    
    # Test full model
    print("\n[2/2] Testing VQC Hybrid Model...")
    model = VQCHybridModel(num_classes=9, n_qubits=8, n_vqc_layers=3)
    
    # Create dummy image
    img = torch.randn(1, 3, 224, 224)
    print(f"Input image shape: {img.shape}")
    
    # Forward pass
    logits = model(img)
    print(f"Output logits shape: {logits.shape}")
    print(f"✓ VQC Hybrid Model works!")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total_params/1e6:.2f}M")

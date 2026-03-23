from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit.circuit.library import ZZFeatureMap

def get_qsvm(num_features=4):
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=3)
    kernel = FidelityQuantumKernel(feature_map=feature_map)

    clf = QSVC(quantum_kernel=kernel)   # ✅ NOT Pegasos
    return clf
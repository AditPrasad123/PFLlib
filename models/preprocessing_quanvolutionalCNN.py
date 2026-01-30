import os
import numpy as np
import cv2
from tqdm import tqdm
import pennylane as qml




# PREPROCESSING SCRIPT FOR QUANVOLUTIONAL CNN

# ----------------------------
# CONFIG
# ----------------------------
# INPUT_DIR = "D:/Capstone/CodeBase/PFLlib/dataset/ISIC2019/train"
# OUTPUT_DIR = "D:/Capstone/CodeBase/PFLlib/dataset/ISIC2019/train_quanv"

INPUT_DIR = "D:/Capstone/CodeBase/PFLlib/dataset/ISIC2019/test"
OUTPUT_DIR = "D:/Capstone/CodeBase/PFLlib/dataset/ISIC2019/test_quanv"


IMAGE_SIZE = 48
PATCH_SIZE = 2
N_QUBITS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Quantum device
# ----------------------------
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def quanv_circuit(inputs):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])

    return qml.expval(qml.PauliZ(0))


# ----------------------------
# Quanvolution on ONE image
# ----------------------------
STRIDE = 4

def quanvolution(image_gray):
    h, w = image_gray.shape
    out_h = (h - PATCH_SIZE) // STRIDE + 1
    out_w = (w - PATCH_SIZE) // STRIDE + 1

    out = np.zeros((out_h, out_w), dtype=np.float32)

    oi = 0
    for i in range(0, h - PATCH_SIZE + 1, STRIDE):
        oj = 0
        for j in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = image_gray[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            patch = patch.flatten()
            patch = patch / 255.0 * np.pi

            out[oi, oj] = quanv_circuit(patch)
            oj += 1
        oi += 1

    return out



# ----------------------------
# MAIN LOOP
# ----------------------------
client_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]

for idx, fname in enumerate(client_files, 1):
    print(f"\n🔹 Processing client file {idx}/{len(client_files)}: {fname}")

    data = np.load(
        os.path.join(INPUT_DIR, fname),
        allow_pickle=True
    )

    client_data = data["data"].item()
    images = client_data["x"]
    labels = client_data["y"]

    quanv_images = []

    for img in tqdm(
        images,
        desc=f"Images in {fname}",
        unit="img",
        leave=True
    ):
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        q_img = quanvolution(img_gray)
        q_img = q_img[..., np.newaxis]

        quanv_images.append(q_img)

    quanv_images = np.stack(quanv_images).astype(np.float32)

    np.savez(
        os.path.join(OUTPUT_DIR, fname),
        data={
            "x": quanv_images,
            "y": labels
        }
    )

    print(f"✅ Saved quanvolved file: {fname}")

print("\n🎉 Quanvolution preprocessing COMPLETE for all clients.")

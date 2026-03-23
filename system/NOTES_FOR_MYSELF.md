PHASE 1: Federated Learning (already working)

Each client:
Image
 → EfficientNet (shared backbone)
 → Projector (128 → 6 features)

Server:
Aggregates backbone (FedBABU)

-------------------------------------

PHASE 2: Personalization

Each client:
 → uses trained backbone
 → extracts 6-dim features
 → trains its OWN QSVM (Qiskit)

-------------------------------------

PHASE 3: Evaluation

Compare:
✔ NN head (FedBABU)
✔ TTFT (optional)
✔ QSVM (your contribution)





Q&A Becoz im too into this!! 

1. Why is QSVM not inside FL?
-> QSVM is not differentiable : it cannot use backdrop, and cannot fit into the training loop

2. Why per-client QSVM?
-> keeps FL decentralized, mathces FedBABU Personalization, stronger paper story

3. What is actually being learned?
-> Backbone: Global Knowledge
   QSVM : Local decision boundary


4. Why we changing the loss to the following:
y_onehot = F.one_hot(y, num_classes=out.shape[1]).float()
loss = torch.nn.MSELoss()(out, y_onehot)
-> Becoz now the ouput is 8 features, not class logits. Therefore CrossEntropy no longer works.
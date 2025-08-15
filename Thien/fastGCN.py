import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, degree
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Manually normalize adjacency matrix (A_hat = D^(-1/2) * A * D^(-1/2))
adj = to_dense_adj(data.edge_index).squeeze(0)  # Shape: [N, N]
deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
D_inv_sqrt = torch.diag(deg_inv_sqrt)
A_hat = D_inv_sqrt @ adj @ D_inv_sqrt
A_hat = A_hat.to(device)

# FastGCN Model
class FastGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(FastGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # Input layer
        self.layers.append(nn.Linear(in_features, hidden_features))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
        # Output layer
        self.layers.append(nn.Linear(hidden_features, out_features))

    def forward(self, X, sampled_nodes_per_layer, A_hat, sampling_probs=None):
        H = X  # Shape: [num_nodes, in_features]
        for l in range(self.num_layers):
            # Get sampled nodes for current layer
            sampled_nodes = sampled_nodes_per_layer[l]
            # Next layer's sampled nodes
            next_sampled_nodes = sampled_nodes_per_layer[l + 1] if l < self.num_layers - 1 else torch.arange(data.num_nodes, device=device)

            # Compute H(l) for sampled nodes
            H_l = F.relu(self.layers[l](H[sampled_nodes]))  # Shape: [sample_size, hidden_features or out_features]

            # Initialize H(l+1) for next layer with correct feature dimension
            out_features = self.layers[l].out_features
            H_next = torch.zeros(data.num_nodes, out_features, device=device)

            # Compute weights for aggregation
            weights = torch.zeros(len(next_sampled_nodes), len(sampled_nodes), device=device)
            for i, v in enumerate(next_sampled_nodes):
                for j, u in enumerate(sampled_nodes):
                    if sampling_probs is None:
                        # Algorithm 1: Uniform sampling
                        weights[i, j] = A_hat[v, u] * data.num_nodes / len(sampled_nodes)
                    else:
                        # Algorithm 2: Importance sampling
                        weights[i, j] = A_hat[v, u] / (sampling_probs[u] * len(sampled_nodes))

            # Compute aggregated features for next_sampled_nodes
            aggregated = torch.matmul(weights, H_l)  # Shape: [len(next_sampled_nodes), out_features]

            # Create new H_next by scattering aggregated values
            H_next = H_next.scatter_(0, next_sampled_nodes.unsqueeze(1).expand(-1, out_features), aggregated)

            H = H_next
            if l < self.num_layers - 1:
                H = F.relu(H)
        
        return H

# Compute sampling probabilities for Algorithm 2 (importance sampling)
def compute_sampling_probs(A_hat):
    # q(u) ∝ ||A_hat(:, u)||_2^2
    norms = torch.norm(A_hat, p=2, dim=0) ** 2
    q = norms / norms.sum()
    return q

# Sample nodes for a layer
def sample_nodes(num_nodes, sample_size, sampling_probs=None):
    if sampling_probs is None:
        # Uniform sampling (Algorithm 1)
        return np.random.choice(num_nodes, size=sample_size, replace=False)
    else:
        # Importance sampling (Algorithm 2)
        return np.random.choice(num_nodes, size=sample_size, replace=False, p=sampling_probs.cpu().numpy())

# Training function
def train(model, data, A_hat, epochs=200, batch_size=128, sample_size=50, lr=0.01, algorithm=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Compute sampling probabilities for Algorithm 2
    sampling_probs = compute_sampling_probs(A_hat) if algorithm == 2 else None

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # Create batches
        perm = torch.randperm(data.num_nodes)
        for i in range(0, data.num_nodes, batch_size):
            optimizer.zero_grad()

            # Get batch indices
            batch_idx = perm[i:i + batch_size]
            batch_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            batch_mask[batch_idx] = True

            # Sample nodes for each layer
            sampled_nodes_per_layer = []
            for l in range(model.num_layers):
                sampled_nodes = sample_nodes(data.num_nodes, sample_size, sampling_probs)
                sampled_nodes = torch.tensor(sampled_nodes, dtype=torch.long, device=device)
                sampled_nodes_per_layer.append(sampled_nodes)
            # Add output layer (batch nodes for training)
            sampled_nodes_per_layer.append(batch_idx)

            # Forward pass
            out = model(data.x, sampled_nodes_per_layer, A_hat, sampling_probs)
            loss = criterion(out[batch_idx], data.y[batch_idx])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / (data.num_nodes // batch_size):.4f}')

# Evaluation function
def evaluate(model, data, A_hat, sample_size=50, algorithm=1):
    model.eval()
    sampling_probs = compute_sampling_probs(A_hat) if algorithm == 2 else None

    with torch.no_grad():
        # Sample nodes for evaluation
        sampled_nodes_per_layer = []
        for l in range(model.num_layers):
            sampled_nodes = sample_nodes(data.num_nodes, sample_size, sampling_probs)
            sampled_nodes = torch.tensor(sampled_nodes, dtype=torch.long, device=device)
            sampled_nodes_per_layer.append(sampled_nodes)
        sampled_nodes_per_layer.append(torch.arange(data.num_nodes, device=device))

        out = model(data.x, sampled_nodes_per_layer, A_hat, sampling_probs)
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    return acc.item()

# Model and training parameters
in_features = dataset.num_features
hidden_features = 16
out_features = dataset.num_classes
num_layers = 2
sample_size = 50

# Initialize model
model = FastGCN(in_features, hidden_features, out_features, num_layers).to(device)

# Train with Algorithm 1 (Uniform Sampling)
print("Training with Algorithm 1 (Uniform Sampling)")
train(model, data, A_hat, epochs=200, batch_size=128, sample_size=sample_size, lr=0.01, algorithm=1)
acc = evaluate(model, data, A_hat, sample_size=sample_size, algorithm=1)
print(f"Test Accuracy (Algorithm 1): {acc:.4f}")

# Reinitialize model for Algorithm 2
model = FastGCN(in_features, hidden_features, out_features, num_layers).to(device)

# Train with Algorithm 2 (Importance Sampling)
print("\nTraining with Algorithm 2 (Importance Sampling)")
train(model, data, A_hat, epochs=200, batch_size=128, sample_size=sample_size, lr=0.01, algorithm=2)
acc = evaluate(model, data, A_hat, sample_size=sample_size, algorithm=2)
print(f"Test Accuracy (Algorithm 2): {acc:.4f}")
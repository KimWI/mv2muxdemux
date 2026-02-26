import numpy as np
import torch
import time
from sklearn.cluster import KMeans

def kmeans_pytorch_opt(data_tensor, n_clusters, init_centroids=None, n_init=3, max_iter=30, tol=1e-4):
    device = data_tensor.device
    N, D = data_tensor.shape
    best_inertia = float('inf')
    best_centroids = None
    
    if init_centroids is not None:
        n_init = 1
        
    for _ in range(n_init):
        if init_centroids is not None:
            centroids = torch.tensor(init_centroids, dtype=torch.float32, device=device)
        else:
            idx = torch.randperm(N)[:n_clusters]
            centroids = data_tensor[idx].clone()
            
        for _ in range(max_iter):
            # Distance computation
            distances = torch.cdist(data_tensor, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # Vectorized Centroid Updates using scatter_add_
            counts = torch.bincount(labels, minlength=n_clusters).float().unsqueeze(1)
            X_sum = torch.zeros(n_clusters, D, device=device, dtype=torch.float32)
            X_sum.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), data_tensor)
            
            # Avoid division by zero
            counts_clamped = torch.clamp(counts, min=1.0)
            new_centroids = X_sum / counts_clamped
            
            # Handle empty clusters
            empty_mask = (counts == 0.0).squeeze(1)
            if empty_mask.any():
                num_empty = empty_mask.sum().item()
                random_idx = torch.randint(0, N, (num_empty,), device=device)
                new_centroids[empty_mask] = data_tensor[random_idx]
                    
            if torch.norm(new_centroids - centroids) < tol:
                centroids = new_centroids
                break
            centroids = new_centroids
            
        distances = torch.cdist(data_tensor, centroids)
        labels = torch.argmin(distances, dim=1)
        inertia = distances[torch.arange(N), labels].pow(2).sum()
        
        if inertia < best_inertia:
            best_inertia, best_centroids = inertia, centroids
            
    return best_centroids.cpu().numpy()

X = np.random.randint(0, 256, (300000, 3)).astype(np.float32)
X_t = torch.tensor(X, device='cuda')

print("--- PyTorch OPT ---")
# warmup
kmeans_pytorch_opt(X_t, 15, n_init=3, max_iter=30)

s = time.time()
kmeans_pytorch_opt(X_t, 15, n_init=3, max_iter=30)
print(f"PyTorch GPU OPT n_init=3: {time.time() - s:.3f} s")

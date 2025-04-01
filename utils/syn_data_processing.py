import os
import numpy as np
import json
from collections import defaultdict

def generate_synthetic_keypoints(num_views=3, num_points=10, noise_std=0.5):
    keypoints = {}
    base_points = np.random.rand(2, num_points) * 100  # 2 x N

    for view in range(1, num_views + 1):
        noise = np.random.randn(2, num_points) * noise_std
        keypoints[f'view{view}'] = base_points + noise

    return keypoints

def generate_random_permutation(n):
    perm = np.random.permutation(n)
    perm_matrix = np.zeros((n, n), dtype=int)
    for i, j in enumerate(perm):
        perm_matrix[i, j] = 1
    return perm, perm_matrix

def generate_relative_permutations(abs_perms):
    P = {}
    views = sorted(abs_perms.keys())
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            Xi = abs_perms[views[i]]
            Xj = abs_perms[views[j]]
            P[f'P{i+1}{j+1}'] = (Xi @ Xj.T).tolist()
    return P
def apply_noise_to_permutation_matrix(P, swap_ratio=0.2):
    P = np.array(P)
    n = P.shape[0]
    num_swaps = int(swap_ratio * n)
    for _ in range(num_swaps):
        i, j = np.random.choice(n, 2, replace=False)
        P[[i, j], :] = P[[j, i], :]
        P[:, [i, j]] = P[:, [j, i]]
    return P.astype(int)

#main function to generate
def generate_synthetic_dataset(num_views=3, num_points=10, swap_ratio=0.2, noise_std=0.5, output_dir="synthetic_data"):
    os.makedirs(output_dir, exist_ok=True)
    keypoints = generate_synthetic_keypoints(num_views, num_points, noise_std)

    # Generate absolute permutations
    abs_perm_dict = {}
    abs_perm_vectors = {}
    for view in range(1, num_views + 1):
        perm_vec, perm_mat = generate_random_permutation(num_points)
        abs_perm_vectors[f'X{view}'] = perm_vec.tolist()
        abs_perm_dict[f'X{view}'] = perm_mat

    # Generate relative permutations
    rel_perms = generate_relative_permutations(abs_perm_dict)

    # Add noise to relative permutations
    noisy_rel_perms = {k: apply_noise_to_permutation_matrix(v, swap_ratio).tolist() for k, v in rel_perms.items()}

    # Save data
    with open(os.path.join(output_dir, "keypoints.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in keypoints.items()}, f, indent=2)
    with open(os.path.join(output_dir, "absolute_permutations.json"), "w") as f:
        json.dump(abs_perm_vectors, f, indent=2)
    with open(os.path.join(output_dir, "relative_permutations.json"), "w") as f:
        json.dump(noisy_rel_perms, f, indent=2)

    print(f"Synthetic dataset saved to '{output_dir}/'")
    return keypoints, abs_perm_dict, noisy_rel_perms

def convert_perm_vectors_to_matrices(abs_perm_vectors):
    matrices = {}
    for k, perm in abs_perm_vectors.items():
        n = len(perm)
        mat = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            mat[i, j] = 1
        matrices[k] = mat
    return matrices

def validate_permutation_matrix(X):
    return (
        np.all(np.sum(X, axis=0) == 1) and
        np.all(np.sum(X, axis=1) == 1) and
        np.all((X == 0) | (X == 1))
    )
def compute_relative_from_absolute(abs_matrices):
    rel = {}
    views = sorted(abs_matrices.keys())
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            key = f'P{i+1}{j+1}'
            rel[key] = (abs_matrices[views[i]] @ abs_matrices[views[j]].T).tolist()
    return rel
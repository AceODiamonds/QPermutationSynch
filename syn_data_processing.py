import os
import numpy as np
import json
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler, BINARY
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd

# ----------------------------
# Synthetic Data Generation
# ----------------------------

def generate_synthetic_keypoints(num_views=3, num_points=10, noise_std=0.5, seed=42):
    np.random.seed(seed)
    keypoints = {}
    base_points = np.random.rand(2, num_points) * 100
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

def generate_relative_permutations(abs_perms, add_symmetric=True):
    P = {}
    views = sorted(abs_perms.keys())
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            Xi = abs_perms[views[i]]
            Xj = abs_perms[views[j]]
            Pij = Xi @ Xj.T
            P[f'P{i+1}{j+1}'] = Pij.tolist()
            if add_symmetric:
                P[f'P{j+1}{i+1}'] = Pij.T.tolist()
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

def generate_synthetic_dataset(num_views=3, num_points=10, swap_ratio=0.2, noise_std=0.5, output_dir="synthetic_data"):
    os.makedirs(output_dir, exist_ok=True)
    keypoints = generate_synthetic_keypoints(num_views, num_points, noise_std)
    abs_perm_dict, abs_perm_vectors = {}, {}
    for view in range(1, num_views + 1):
        perm_vec, perm_mat = generate_random_permutation(num_points)
        abs_perm_vectors[f'X{view}'] = perm_vec.tolist()
        abs_perm_dict[f'X{view}'] = perm_mat
    rel_perms = generate_relative_permutations(abs_perm_dict)
    noisy_rel_perms = {k: apply_noise_to_permutation_matrix(v, swap_ratio).tolist() for k, v in rel_perms.items()}
    with open(os.path.join(output_dir, "keypoints.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in keypoints.items()}, f, indent=2)
    with open(os.path.join(output_dir, "absolute_permutations.json"), "w") as f:
        json.dump(abs_perm_vectors, f, indent=2)
    with open(os.path.join(output_dir, "relative_permutations.json"), "w") as f:
        json.dump(noisy_rel_perms, f, indent=2)
    print(f"[✓] Synthetic dataset saved to '{output_dir}/'")
    return keypoints, abs_perm_dict, noisy_rel_perms

# ----------------------------
# QUBO Construction & Solver
# ----------------------------

def qubo_form_maker_fixed_gauge(P, num_views, num_keypoints, penalty=2.5):
    m, n = num_views, num_keypoints
    N = n * n
    m_opt = m - 1
    num_vars = m_opt * N
    Q_prime = np.zeros((num_vars, num_vars))
    s_opt = np.zeros(num_vars)
    x1 = np.zeros(N)
    for k in range(n): x1[k*n + k] = 1

    for key, p_ij in P.items():
        i, j = int(key[1])-1, int(key[2])-1
        block = np.kron(np.eye(n), np.array(p_ij))
        if i > 0 and j > 0:
            Q_prime[(i-1)*N:i*N, (j-1)*N:j*N] -= block
        elif i == 0 and j > 0:
            s_opt[(j-1)*N:j*N] += -x1.T @ block
        elif i > 0 and j == 0:
            s_opt[(i-1)*N:i*N] += -(block @ x1)

    A = np.zeros((2*m_opt*n, num_vars))
    b = np.ones(2*m_opt*n)
    row = 0
    for v in range(m_opt):
        offset = v * N
        for r in range(n): A[row, offset + r*n : offset + (r+1)*n] = 1; row += 1
        for c in range(n): A[row, offset + c::n] = 1; row += 1

    Q = Q_prime + penalty * A.T @ A
    s = s_opt - 2 * penalty * A.T @ b
    print("created the Q and s for this run")
    return Q, s

def solve_qubo(Q, s, reads=1000):
    bqm = BinaryQuadraticModel(s, Q, 0.0, vartype=BINARY)
    sampler = SimulatedAnnealingSampler()
    res = sampler.sample(bqm, num_reads=reads)
    print("This QUBO run was solved.")
    return res.first.sample, res.first.energy

def decode_solution(sample, num_views, num_keypoints):
    N = num_keypoints * num_keypoints
    decoded = {"X1": np.eye(num_keypoints, dtype=int)}
    flat = [sample[k] for k in sorted(sample.keys())]
    for v in range(2, num_views + 1):
        start = (v-2)*N
        decoded[f'X{v}'] = np.array(flat[start:start+N]).reshape(num_keypoints, num_keypoints)
    print('Decoded the solution for this run')
    return decoded

def permutation_accuracy(true_perm, pred_perm):
    """
    Calculates accuracy as the ratio of correctly matched keypoints.
    """
    return np.sum(true_perm == pred_perm) / true_perm.size

# ----------------------------
# Full Pipeline
# ----------------------------

def run_pipeline(): # without post data processing
    keypoints, abs_perm, P = generate_synthetic_dataset(num_views=4, num_points=4, swap_ratio=0.2)
    Q, s = qubo_form_maker_fixed_gauge(P, num_views=4, num_keypoints=4)
    sample, energy = solve_qubo(Q, s)
    decoded = decode_solution(sample, num_views=4, num_keypoints=4)

    print("\nQUBO done. The permutation matrices are:")
    for k, v in decoded.items():
        print(f"{k}:\n{np.array(v)}\n")

def benchmark_pipeline(runs=5, views_list=[3, 4], keypoints_list=[4, 5], swap_ratio=0.2, noise_std=0.5):
    results = []
    output_dir = "synth_data_results"
    os.makedirs(output_dir, exist_ok=True)

    for num_views in views_list:
        for num_keypoints in keypoints_list:
            for run in range(runs):
                start_time = time.time()

                # Generate data and the run pipeline
                _, abs_perm, P = generate_synthetic_dataset(num_views=num_views, num_points=num_keypoints,
                                                            swap_ratio=swap_ratio, noise_std=noise_std)
                Q, s = qubo_form_maker_fixed_gauge(P, num_views=num_views, num_keypoints=num_keypoints)
                sample, energy = solve_qubo(Q, s)
                decoded = decode_solution(sample, num_views=num_views, num_keypoints=num_keypoints)

                # Calculate accuracy for each view
                for v in range(2, num_views + 1):
                    true_perm = abs_perm[f'X{v}']
                    pred_perm = decoded[f'X{v}']
                    acc = permutation_accuracy(np.array(true_perm), np.array(pred_perm))
                    results.append({
                        'run': run,
                        'views': num_views,
                        'keypoints': num_keypoints,
                        'view': v,
                        'accuracy': acc,
                        'energy': energy,
                        'time_sec': time.time() - start_time
                    })

    df = pd.DataFrame(results)

    # Save graphs with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for metric in ['accuracy', 'energy', 'time_sec']:
        plt.figure()
        for key, group in df.groupby(['views', 'keypoints']):
            label = f"Views={key[0]}, Keypoints={key[1]}"
            plt.plot(group['run'], group[metric], marker='o', label=label)
        plt.title(f"{metric.capitalize()} vs Run")
        plt.xlabel("Run")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"benchmark_{metric}_{timestamp}.png"))
        plt.close()

    print(f"benchmarking completed. Results and graphs saved with timestamp {timestamp}")
    return df

if __name__ == "__main__":
    benchmark_pipeline(runs=3, views_list=[3], keypoints_list=[4, 5])
